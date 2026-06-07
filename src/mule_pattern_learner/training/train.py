"""Train the nnPU mule-detection GNN.

Production training path, end to end: seeds and PU targets are read from the
GRAPH (the is_train / is_val / is_test flags and pu_label the masking step wrote
onto Accounts) via get_split_accounts, the model trains with early stopping on
validation Proxy AUC (computed each epoch on a fixed seeded val subsample for
speed), and the best checkpoint is saved to models/.

One required argument, --estimated-mules: the class prior pi (true mule
fraction) drives nnPU's positive_risk term, and pi cannot be derived -- in
production the true number of mules is unknown. So the operator supplies their
best estimate of the TOTAL mule count, and pi = estimated_mules / total_accounts
is computed against the graph's account count (which IS derived). On the
synthetic data the true count is ~216; passing that emulates "we guessed right".
Everything else (max_bins, edge_dim, reference_epoch_s) is derived from the
graph, never hardcoded.

    mule-train --estimated-mules 216

Everything here uses pu_label only -- the labels the model trains on, which
exist on real data too -- so nothing depends on knowing the true (hidden) mules.
Measuring generalization to hidden mules is a SEPARATE, synthetic-only step
(scripts/.../evaluate_hidden.py), run after training against the answer-key
parquet; it is deliberately not part of this training loop.

Watch the live per-epoch line:
  * val AP at epoch 0 near the base rate (else neighborhood leakage)
  * train loss falling while val AP stalls -> memorizing the few positives;
    early stopping keeps the best-val-AP checkpoint regardless.
"""

from collections.abc import Iterable, Iterator
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import cast

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from mule_pattern_learner.device import select_device
from mule_pattern_learner.features.nodes import (
    build_account_features,
    normalizer_from_features,
)
from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.fetch import fetch_account_vertices
from mule_pattern_learner.pyg.model import MulePatternModel
from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.derivation import (
    derive_reference_epoch_s,
    derive_temporal_spec,
)
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.training.loop import TrainConfig, fit
from mule_pattern_learner.training.seeds import SeedPool, epoch_batches
from mule_pattern_learner.training.seeds_source import fetch_split_seeds

_MASKS_DIR = Path("data/masks")
_COL_ACCOUNT_ID = "account_id"

_MODELS_DIR = Path("models")
_ACCOUNT_FEATURES = 31

# Seed-batch composition and training schedule. These are project defaults
# (in-code constants, not CLI args). Each seed-batch is one TigerGraph sampling
# round-trip (~the per-batch cost), so batch size trades round-trips against
# per-query work: larger batches mean far fewer queries per epoch.
# positives_per_batch holds the per-batch positive density (~6.25%) that nnPU's
# positive_risk term needs, scaled with batch size.
#
# The one exception is the class prior pi, which is NOT a constant here: in
# production it is an assumption that cannot be derived (we never know the true
# number of mules), so it is supplied per run as --estimated-mules and turned
# into pi against the graph's account count. See _parse_args / _resolve_prior.
_BATCH_SIZE = 512
# Kept at 1/16 of the batch (6.25% positives) so halving _BATCH_SIZE changes
# only peak memory, not the per-batch positive fraction.
_POSITIVES_PER_BATCH = 32
_MAX_EPOCHS = 30
_PATIENCE = 5
_EVAL_K = 100
_RNG_SEED = 1337
# Validation cost is dominated by the unlabeled accounts, not the few positives,
# so per-epoch validation scores a fixed seeded subsample: ALL revealed positives
# (the rare-class signal, kept whole to avoid inflating metric variance) plus this
# many unlabeled. Full-population scoring is left to the separate hidden-mule eval.
_VAL_UNLABELED_CAP = 10_000
# Train accounts are fetched in chunks to fit the normalizer without pulling all
# ~130k feature rows in a single request.
_FEATURE_FETCH_CHUNK = 5_000
_VERTEX_ACCOUNT = "Account"


@dataclass(frozen=True, slots=True)
class _Args:
    estimated_mules: int


def _parse_args() -> _Args:
    parser = argparse.ArgumentParser(
        prog="mule-train",
        description=(
            "Train the nnPU mule-detection GNN. Everything except the class "
            "prior is derived from the graph. The prior pi (true mule fraction) "
            "cannot be derived -- in production the true number of mules is "
            "unknown -- so you supply your best estimate of the TOTAL mule count "
            "and pi is computed against the graph's account count."
        ),
    )
    _ = parser.add_argument(
        "--estimated-mules",
        type=int,
        required=True,
        metavar="N",
        help=(
            "Estimated TOTAL number of mules in the account population (your "
            "assumption; not the count of revealed/labelled mules). pi = N / "
            "total_accounts. On the synthetic data the true count is ~216."
        ),
    )
    ns = parser.parse_args()
    return _Args(estimated_mules=cast(int, ns.estimated_mules))


def _resolve_prior(client: Client, estimated_mules: int) -> float:
    # Turn the operator's estimated total mule count into the nnPU class prior
    # pi = estimated_mules / total_accounts. total_accounts is derived from the
    # graph (realtime=True forces an accurate recount rather than a cached, up-
    # to-30s-stale figure). pi must land in (0, 1) or nnPU is undefined, so an
    # out-of-range estimate is rejected here with a message naming both numbers.
    if estimated_mules < 1:
        raise ValueError(f"--estimated-mules must be >= 1, got {estimated_mules}")
    # getVertexCount returns int for a single vertex type (dict only for "*" or a
    # list of types), so narrow the int | dict union to the int we expect here.
    raw_count = client.conn.getVertexCount(_VERTEX_ACCOUNT, realtime=True)
    if not isinstance(raw_count, int):
        raise TypeError(f"expected int account count, got {type(raw_count).__name__}")
    total_accounts = raw_count
    if total_accounts < 1:
        raise ValueError(f"graph reports {total_accounts} {_VERTEX_ACCOUNT} vertices; cannot train")
    if estimated_mules >= total_accounts:
        raise ValueError(
            f"--estimated-mules ({estimated_mules}) must be < total accounts "
            + f"({total_accounts}); pi = estimated_mules / total_accounts must be in (0, 1)"
        )
    prior = estimated_mules / total_accounts
    print(
        f"prior: pi={prior:.6f} (estimated_mules={estimated_mules} / "
        + f"total_accounts={total_accounts})"
    )
    return prior


def main() -> None:
    args = _parse_args()
    client = Client(Settings())
    print(f"connected: {client.graphname}")
    backend = TigerGraphRemoteBackend(client)
    mapper = backend.mapper
    device = select_device()
    print(f"device: {device}")

    # pi is supplied per run (not derivable): estimated total mules / accounts.
    prior = _resolve_prior(client, args.estimated_mules)

    # derived from the graph (single source of truth)
    spec = derive_temporal_spec(client)
    max_bins = spec.max_bins
    edge_dim = spec.edge_dim
    reference_epoch_s = derive_reference_epoch_s(client)
    print(
        f"derived: max_bins={max_bins} edge_dim={edge_dim} "
        + f"reference_epoch_s={reference_epoch_s:.0f}"
    )

    # ── seeds + PU targets FROM THE GRAPH (production path) ──
    train_seeds = fetch_split_seeds(client, "train")
    val_seeds = fetch_split_seeds(client, "val")
    pu_label_of = dict(train_seeds.pu_label_of)
    pu_label_of.update(val_seeds.pu_label_of)
    train_pool = SeedPool(
        positives=tuple(a for a, y in train_seeds.pu_label_of.items() if y == 1),
        unlabeled=tuple(a for a, y in train_seeds.pu_label_of.items() if y == 0),
    )
    print(
        f"graph seeds: train pos={train_pool.num_positives} "
        + f"unl={train_pool.num_unlabeled} val pos={val_seeds.num_positives} "
        + f"val={len(val_seeds.account_ids)}"
    )

    # Fail fast, BEFORE the first (multi-hour) epoch: nnPU needs revealed
    # positives in train (its positive_risk term), and PU model selection (Proxy
    # AUC early-stop) needs at least one revealed positive in val to rank against
    # the unlabeled. With so few revealed mules, a party-grouped split can leave
    # val with none -- which previously surfaced only as a crash at the end of
    # epoch 0's validation, after hours of wasted compute. Refuse up front.
    if train_pool.num_positives < 1:
        raise SystemExit(
            "train split has 0 revealed positives (pu_label==1); nnPU cannot train. "
            "Re-mask (higher reveal prevalence) or fix the split."
        )
    if val_seeds.num_positives < 1:
        raise SystemExit(
            "val split has 0 revealed positives (pu_label==1); Proxy-AUC model "
            "selection is undefined without a positive to rank. Re-mask (higher "
            "reveal prevalence) or adjust the split so val receives some positives."
        )

    # Shrink the per-epoch validation set: keep every revealed positive, take a
    # fixed seeded sample of the unlabeled up to _VAL_UNLABELED_CAP. Drawn ONCE
    # here (not per epoch) so the val set is identical across epochs and the only
    # thing moving PAUC is the model, not the sample. Keeping all positives means
    # this does not worsen the rare-class variance that dominates the metric; it
    # only removes redundant unlabeled scoring, which is pure cost. The full
    # population is still scored by the separate hidden-mule evaluation.
    val_positives = [a for a in val_seeds.account_ids if pu_label_of.get(a) == 1]
    val_unlabeled = [a for a in val_seeds.account_ids if pu_label_of.get(a) != 1]
    if len(val_unlabeled) > _VAL_UNLABELED_CAP:
        val_unlabeled = random.Random(_RNG_SEED).sample(val_unlabeled, _VAL_UNLABELED_CAP)
    val_seed_ids = tuple(val_positives) + tuple(val_unlabeled)
    print(
        f"val subsample: {len(val_positives)} positives + {len(val_unlabeled)} unlabeled "
        + f"= {len(val_seed_ids)} (full val was {len(val_seeds.account_ids)})"
    )

    # Fit feature standardization on the TRAIN split only (leakage-safe), using
    # the same log/symlog transforms the loaders apply, then reuse it unchanged
    # for val/test. Computed here in Python (not GSQL) so it sees the post-
    # transform values and never mixes in val/test statistics. Saved with the
    # checkpoint so scoring reproduces the exact training-time standardization.
    train_ids = tuple(train_seeds.account_ids)
    feature_rows: list[Tensor] = []
    for start in range(0, len(train_ids), _FEATURE_FETCH_CHUNK):
        chunk = list(train_ids[start : start + _FEATURE_FETCH_CHUNK])
        vertices = fetch_account_vertices(client, chunk)
        feature_rows.append(build_account_features(vertices).feats)
    train_features = torch.cat(feature_rows, dim=0)
    normalizer = normalizer_from_features(train_features)
    print(
        f"feature normalizer fit on {train_features.shape[0]} train accounts "
        + f"({train_features.shape[1]} features standardized)"
    )

    fanout = NeighborFanout()

    def mapper_to_ids(int_ids: list[int]) -> list[str]:
        return mapper.to_strings("Account", int_ids)

    # TRAIN loader: strict-inductive, excludes val AND test from neighborhoods.
    def make_train_loader() -> Iterator[HeteroData]:
        for seed_batch in epoch_batches(
            train_pool,
            batch_size=_BATCH_SIZE,
            positives_per_batch=_POSITIVES_PER_BATCH,
            seed=_RNG_SEED,
        ):
            loader = backend.make_loader(
                seed_ids=seed_batch,
                reference_epoch_s=reference_epoch_s,
                max_bins=max_bins,
                fanout=fanout,
                batch_size=len(seed_batch),
                shuffle=False,
                allow_val=False,
                allow_test=False,
                normalizer=normalizer,
            )
            yield from cast(Iterable[HeteroData], loader)

    # VAL loader: may use train neighbors (seen), never test.
    def make_val_loader() -> Iterator[HeteroData]:
        loader = backend.make_loader(
            seed_ids=val_seed_ids,
            reference_epoch_s=reference_epoch_s,
            max_bins=max_bins,
            fanout=fanout,
            batch_size=_BATCH_SIZE,
            shuffle=False,
            allow_val=True,
            allow_test=False,
            normalizer=normalizer,
        )
        yield from cast(Iterable[HeteroData], loader)

    model = MulePatternModel(account_in_dim=_ACCOUNT_FEATURES, edge_dim=edge_dim)
    config = TrainConfig(prior=prior, max_epochs=_MAX_EPOCHS, patience=_PATIENCE, eval_k=_EVAL_K)

    # per-epoch batch counts, for the progress-bar ETA (loaders are lazy
    # generators with no len): train consumes the unlabeled pool once per epoch
    # at (batch_size - positives_per_batch) unlabeled per batch; val covers all
    # val seeds at batch_size per batch.
    unlabeled_per_batch = _BATCH_SIZE - _POSITIVES_PER_BATCH
    train_batches = max(1, math.ceil(train_pool.num_unlabeled / unlabeled_per_batch))
    val_batches = max(1, math.ceil(len(val_seed_ids) / _BATCH_SIZE))

    print("=" * 70)
    print("TRAINING (nnPU, strict-inductive, early-stop on val Proxy AUC)")
    print("=" * 70)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _MODELS_DIR / f"mule_model_seed{_RNG_SEED}.pt"

    # save the best checkpoint the moment each new best epoch is found, so an
    # interrupted long run still leaves the best model on disk. A fixed filename
    # means each new best overwrites the previous one.
    def save_best(state: dict[str, Tensor], epoch: int, val_pauc: float) -> None:
        checkpoint: dict[str, object] = {
            "model_state_dict": state,
            "best_epoch": epoch,
            "best_val_pauc": val_pauc,
            "account_in_dim": _ACCOUNT_FEATURES,
            "edge_dim": edge_dim,
            "prior": prior,
            "reference_epoch_s": reference_epoch_s,
            "max_bins": max_bins,
            "feature_mean": normalizer.mean,
            "feature_std": normalizer.std,
        }
        torch.save(checkpoint, checkpoint_path)
        print(
            f"  saved best (epoch {epoch}, val PAUC {val_pauc:.4f}) -> {checkpoint_path}",
            flush=True,
        )

    result = fit(
        model=model,
        make_train_loader=make_train_loader,
        make_val_loader=make_val_loader,
        pu_label_of=pu_label_of,
        mapper_to_ids=mapper_to_ids,
        config=config,
        train_batches=train_batches,
        val_batches=val_batches,
        device=device,
        on_best=save_best,
    )

    print(
        f"{'epoch':>5} {'train_loss':>12} {'val_PAUC':>10} {'val_AP':>10} {'val_P@k':>10} {'best':>6}"
    )
    for r in result.reports:
        marker = "*" if r.is_best else ""
        print(
            f"{r.epoch:>5} {r.train_loss:>12.4f} {r.val_roc_auc:>10.4f} "
            + f"{r.val_average_precision:>10.4f} {r.val_precision_at_k:>10.4f} {marker:>6}"
        )
    print(f"\nbest epoch {result.best_epoch}: val PAUC = {result.best_val_pauc:.4f}")
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
