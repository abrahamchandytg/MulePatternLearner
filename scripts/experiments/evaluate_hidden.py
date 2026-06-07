"""Synthetic-only evaluation: does the trained model find HIDDEN mules?

SYNTHETIC DATA ONLY. This step needs the answer key (true_label / bucket) from
the masking parquet, which does not exist in production -- so this is separate
from training, run after it. It loads the latest checkpoint, scores a split's
accounts, and reports generalization to mules that were never labelled.

No arguments. Loads the most recent models/mule_model_*.pt, scores the TEST
split (the held-out dark rings -- the hardest generalization test) against the
parquet answer key, and prints hidden-recall@k plus AP/AUC against true labels.

    python scripts/.../evaluate_hidden.py
"""

from collections.abc import Iterable
import math
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from tqdm import tqdm

from mule_pattern_learner.features.nodes import FeatureNormalizer
from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.model import MulePatternModel
from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.training.metrics import evaluate_hidden
from mule_pattern_learner.training.seeds_source import fetch_split_seeds

_MASKS_DIR = Path("data/masks")
_MODELS_DIR = Path("models")
_COL_ACCOUNT_ID = "account_id"
_COL_TRUE_LABEL = "true_label"
_COL_BUCKET = "bucket"

# Which split to evaluate generalization on. test holds the dark rings (mules
# whose whole ring was hidden) -- the strongest test of finding unseen mules.
_EVAL_SPLIT = "test"
_EVAL_K = 100
_BATCH_SIZE = 1024


_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")


class _NodeStore(Protocol):
    n_id: Tensor
    x: Tensor
    batch_size: int


class _EdgeStore(Protocol):
    edge_index: Tensor
    edge_attr: Tensor


def _node_store(batch: HeteroData, key: NodeType) -> _NodeStore:
    return cast(_NodeStore, cast(object, batch[key]))


def _edge_store(batch: HeteroData, key: EdgeType) -> _EdgeStore:
    return cast(_EdgeStore, cast(object, batch[key]))


def _find_latest_checkpoint(models_dir: Path) -> Path:
    candidates = sorted(
        models_dir.glob("mule_model_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no mule_model_*.pt in {models_dir}; train first.")
    return candidates[0]


def _find_eval_parquet(masks_dir: Path) -> Path:
    candidates = sorted(
        masks_dir.glob("pu_labels_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no pu_labels_*.parquet in {masks_dir}.")
    return candidates[0]


def _labels(frame: pd.DataFrame, column: str) -> dict[str, int]:
    ids: list[str] = frame[_COL_ACCOUNT_ID].astype(str).tolist()
    values: list[int] = frame[column].astype(int).tolist()
    return dict(zip(ids, values))


def main() -> None:
    client = Client(Settings())
    print(f"connected: {client.graphname}")
    backend = TigerGraphRemoteBackend(client)
    mapper = backend.mapper

    # load the checkpoint + its metadata (dims/recency it was trained with)
    ckpt_path = _find_latest_checkpoint(_MODELS_DIR)
    checkpoint = cast(dict[str, object], torch.load(ckpt_path, weights_only=False))
    edge_dim = cast(int, checkpoint["edge_dim"])
    max_bins = cast(int, checkpoint["max_bins"])
    reference_epoch_s = cast(float, checkpoint["reference_epoch_s"])
    account_in_dim = cast(int, checkpoint["account_in_dim"])
    state_dict = cast("dict[str, torch.Tensor]", checkpoint["model_state_dict"])
    # Rebuild the training-fit feature standardization so eval scores features
    # exactly as training saw them.
    normalizer = FeatureNormalizer(
        mean=cast(torch.Tensor, checkpoint["feature_mean"]),
        std=cast(torch.Tensor, checkpoint["feature_std"]),
    )
    print(
        f"checkpoint: {ckpt_path.name} (best_val_pauc={cast(float, checkpoint['best_val_pauc']):.4f})"
    )

    model = MulePatternModel(account_in_dim=account_in_dim, edge_dim=edge_dim)
    _ = model.load_state_dict(state_dict)
    _ = model.eval()

    # seeds for the eval split, from the graph
    split_seeds = fetch_split_seeds(client, _EVAL_SPLIT)
    seed_ids = split_seeds.account_ids
    print(f"{_EVAL_SPLIT} split: {len(seed_ids)} accounts")

    # answer key from the parquet (synthetic only)
    eval_path = _find_eval_parquet(_MASKS_DIR)
    frame = pd.read_parquet(eval_path)
    true_label_of = _labels(frame, _COL_TRUE_LABEL)
    bucket_of = _labels(frame, _COL_BUCKET)
    print(f"answer key: {eval_path.name}")

    fanout = NeighborFanout()

    def mapper_to_ids(int_ids: list[int]) -> list[str]:
        return mapper.to_strings("Account", int_ids)

    # test-time sampling may use the whole graph (allow_val=True, allow_test=True)
    loader = backend.make_loader(
        seed_ids=seed_ids,
        reference_epoch_s=reference_epoch_s,
        max_bins=max_bins,
        fanout=fanout,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        allow_val=True,
        allow_test=True,
        normalizer=normalizer,
    )

    scores: list[float] = []
    seen_ids: list[str] = []
    eval_batches = max(1, math.ceil(len(seed_ids) / _BATCH_SIZE))
    bar = tqdm(
        cast(Iterable[HeteroData], loader),
        total=eval_batches,
        unit="batch",
        miniters=1,
        desc="  eval",
    )
    with torch.no_grad():
        for batch in bar:
            account = _node_store(batch, "Account")
            bsize = int(account.batch_size)
            n_id = cast("list[int]", account.n_id[:bsize].tolist())
            seeds = mapper_to_ids(n_id)
            x_dict: dict[NodeType, Tensor] = {"Account": account.x}
            node_counts: dict[NodeType, int] = {
                nt: int(_node_store(batch, nt).n_id.shape[0]) for nt in batch.node_types
            }
            edge_index_dict: dict[EdgeType, Tensor] = {
                et: _edge_store(batch, et).edge_index for et in batch.edge_types
            }
            edge_attr_dict: dict[EdgeType, Tensor] = {
                _HAS_PAID: _edge_store(batch, _HAS_PAID).edge_attr
            }
            logits = cast(Tensor, model(x_dict, edge_index_dict, edge_attr_dict, node_counts))
            probs = cast("list[float]", torch.sigmoid(logits[:bsize]).tolist())
            scores.extend(probs)
            seen_ids.extend(seeds)

    scores_arr: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    true_arr: NDArray[np.int64] = np.asarray([true_label_of[s] for s in seen_ids], dtype=np.int64)
    bucket_arr: NDArray[np.int64] = np.asarray([bucket_of[s] for s in seen_ids], dtype=np.int64)

    result = evaluate_hidden(scores_arr, true_arr, bucket_arr, k=_EVAL_K)
    print("\n" + "=" * 60)
    print(f"HIDDEN-MULE GENERALIZATION on the {_EVAL_SPLIT} split (synthetic)")
    print("=" * 60)
    print(f"  accounts scored        : {len(seen_ids)}")
    print(f"  true mules             : {result.num_true_positives}")
    print(f"  hidden mules           : {result.num_hidden_positives}")
    print(f"  hidden_recall@{result.k:<4}    : {result.hidden_recall_at_k:.4f}")
    print(f"  AP (vs true labels)    : {result.average_precision_true:.4f}")
    print(f"  AUC (vs true labels)   : {result.roc_auc_true:.4f}")
    print("=" * 60)
    print("hidden_recall@k is the headline: of mules the model was NEVER told")
    print("about, the fraction it ranked in the top-k. High = it learned the")
    print("pattern, not the few labelled examples.")


if __name__ == "__main__":
    main()
