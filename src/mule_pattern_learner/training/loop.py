from collections.abc import Callable, Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from tqdm import tqdm

from mule_pattern_learner.device import select_device
from mule_pattern_learner.training.loss import NonNegativePULoss
from mule_pattern_learner.training.metrics import ValScores, evaluate_ranking

_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")
_ACCOUNT: NodeType = "Account"

# A batch transform hook (e.g. for augmentation); identity by default.
BatchTransform = Callable[[HeteroData], HeteroData]


def _identity(batch: HeteroData) -> HeteroData:
    return batch


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """
    Training-loop hyperparameters.

    prior:        class prior pi for the nnPU loss (true mule fraction).
    max_epochs:   hard cap on epochs.
    patience:     early-stopping patience (epochs without val-PAUC improvement).
    lr / weight_decay: AdamW settings (weight_decay is the decoupled L2 lever).
    eval_k:       cutoff for precision@k in validation.
    min_delta:    minimum val-PAUC gain to count as a significant improvement.
    """

    prior: float
    max_epochs: int = 50
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_k: int = 100
    min_delta: float = 1e-4


class EarlyStopper:
    """
    Tracks the best validation score and decides when to stop.

    A higher score is better (we use validation Proxy AUC). Two distinct
    judgments, deliberately separated:

      * is_best (the return of update): a STRICT improvement, score > best so
        far. Any genuine gain, however small, makes these the best weights, so
        the caller should save them. Using strict comparison here means the
        saved checkpoint is always the true best-val-PAUC model.
      * patience: only a SIGNIFICANT improvement -- beating the last significant
        score by at least min_delta -- resets the no-improve counter. Sub-delta
        gains are treated as noise for stopping purposes, so training still
        halts once real progress stalls.

    Keeping these apart fixes the trap where a tiny-but-real improvement was
    neither saved nor counted as progress: now it is saved (is_best) even though
    it does not reset patience.
    """

    _patience: int
    _min_delta: float
    _best: float
    _significant_best: float
    _bad_epochs: int
    best_epoch: int

    def __init__(self, patience: int, min_delta: float) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._best = float("-inf")
        self._significant_best = float("-inf")
        self._bad_epochs = 0
        self.best_epoch = -1

    @property
    def best(self) -> float:
        return self._best

    @property
    def bad_epochs(self) -> int:
        """Epochs since the last min_delta-significant improvement (patience progress)."""
        return self._bad_epochs

    def update(self, score: float, epoch: int) -> bool:
        """Record an epoch's score; return True if it was a new (strict) best.

        A True result means save: score beat the best seen so far. Patience is
        managed separately -- it only resets on a min_delta-significant gain --
        so a sub-delta improvement returns True (save it) yet still advances the
        no-improve counter toward early stopping.
        """
        is_best = score > self._best
        if is_best:
            self._best = score
            self.best_epoch = epoch
        if score > self._significant_best + self._min_delta:
            self._significant_best = score
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
        return is_best

    @property
    def should_stop(self) -> bool:
        return self._bad_epochs >= self._patience


class _NodeStore(Protocol):
    n_id: Tensor
    x: Tensor


class _EdgeStore(Protocol):
    edge_index: Tensor
    edge_attr: Tensor


def _node_store(batch: HeteroData, key: NodeType) -> _NodeStore:
    return cast(_NodeStore, cast(object, batch[key]))


def _edge_store(batch: HeteroData, key: EdgeType) -> _EdgeStore:
    return cast(_EdgeStore, cast(object, batch[key]))


def _forward(model: Module, batch: HeteroData, device: torch.device) -> Tensor:
    # assemble the model's inputs from a HeteroData batch, moving every tensor
    # onto the compute device, then run the forward
    x_dict: dict[NodeType, Tensor] = {_ACCOUNT: _node_store(batch, _ACCOUNT).x.to(device)}
    node_counts: dict[NodeType, int] = {
        ntype: int(_node_store(batch, ntype).n_id.shape[0]) for ntype in batch.node_types
    }
    edge_index_dict: dict[EdgeType, Tensor] = {
        etype: _edge_store(batch, etype).edge_index.to(device) for etype in batch.edge_types
    }
    edge_attr_dict: dict[EdgeType, Tensor] = {
        _HAS_PAID: _edge_store(batch, _HAS_PAID).edge_attr.to(device)
    }
    return cast(Tensor, model(x_dict, edge_index_dict, edge_attr_dict, node_counts))


def _seed_ids(batch: HeteroData, mapper_to_ids: Callable[[list[int]], list[str]]) -> list[str]:
    # the first batch_size Account nodes are exactly the seeds (PyG guarantee)
    bsize = int(batch[_ACCOUNT].batch_size)  # pyright: ignore[reportAny]
    seed_int = cast(list[int], _node_store(batch, _ACCOUNT).n_id[:bsize].tolist())
    return mapper_to_ids(seed_int)


def _targets(seed_ids: Sequence[str], label_of: Mapping[str, int], device: torch.device) -> Tensor:
    return torch.tensor([label_of[s] for s in seed_ids], dtype=torch.long, device=device)


def train_epoch(
    model: Module,
    loader: Iterator[HeteroData],
    loss_fn: NonNegativePULoss,
    optimizer: Optimizer,
    pu_label_of: Mapping[str, int],
    mapper_to_ids: Callable[[list[int]], list[str]],
    batch_transform: BatchTransform = _identity,
    total_batches: int | None = None,
    epoch: int = 0,
    device: torch.device | None = None,
) -> float:
    """
    One training epoch. Returns the mean training loss over batches.

    For each batch: forward the whole sampled neighborhood, slice to the seed
    accounts (logits[:batch_size]), look up their pu_label targets, compute the
    nnPU loss on the seeds only, and backprop. Neighbors only inform embeddings;
    they never contribute to the loss.

    total_batches, if given, drives the progress bar's ETA (the loader is a lazy
    generator with no len, so the count must be passed in). miniters=1 because
    per-batch latency is erratic (each batch is a TigerGraph round-trip).
    """
    dev = device if device is not None else torch.device("cpu")
    _ = model.train()
    total = 0.0
    count = 0
    bar = tqdm(loader, total=total_batches, unit="batch", miniters=1, desc=f"train e{epoch}")
    for raw in bar:
        batch = batch_transform(raw)
        logits = _forward(model, batch, dev)
        seeds = _seed_ids(batch, mapper_to_ids)
        seed_logits = logits[: len(seeds)]
        targets = _targets(seeds, pu_label_of, dev)

        train_loss, _ = cast("tuple[Tensor, Tensor]", loss_fn(seed_logits, targets))
        optimizer.zero_grad()
        _ = train_loss.backward()
        _ = optimizer.step()

        total += float(train_loss.item())
        count += 1
        bar.set_postfix(loss=f"{total / count:.4f}")
    return total / max(count, 1)


def validate(
    model: Module,
    loader: Iterator[HeteroData],
    pu_label_of: Mapping[str, int],
    mapper_to_ids: Callable[[list[int]], list[str]],
    eval_k: int,
    total_batches: int | None = None,
    epoch: int = 0,
    device: torch.device | None = None,
) -> ValScores:
    """
    Score the model over the validation seeds and return ranking metrics.

    Collects per-seed scores (sigmoid of the logit) and the seed's pu_label
    across all val batches, then evaluates ranking quality against pu_label.
    This uses only the labels the model also trains on, so it is production-valid.
    The loader must use the
    validation sampling regime (allow_val=True, allow_test=False) so no test
    account leaks into a val neighborhood.
    """
    dev = device if device is not None else torch.device("cpu")
    _ = model.eval()
    scores: list[float] = []
    labels: list[int] = []
    raw_logits: list[float] = []
    bar = tqdm(loader, total=total_batches, unit="batch", miniters=1, desc=f"  val e{epoch}")
    with torch.no_grad():
        for batch in bar:
            logits = _forward(model, batch, dev)
            seeds = _seed_ids(batch, mapper_to_ids)
            seed_logits = logits[: len(seeds)]
            probs = cast(list[float], torch.sigmoid(seed_logits).tolist())
            scores.extend(probs)
            raw_logits.extend(cast(list[float], seed_logits.tolist()))
            labels.extend(pu_label_of[s] for s in seeds)

    scores_arr: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    labels_arr: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)

    logits_arr: NDArray[np.float64] = np.asarray(raw_logits, dtype=np.float64)
    pos_mask = labels_arr == 1
    pos_logits = logits_arr[pos_mask]
    unl_logits = logits_arr[~pos_mask]
    pos_mean = float(pos_logits.mean()) if pos_logits.size else float("nan")
    unl_mean = float(unl_logits.mean()) if unl_logits.size else float("nan")
    print(
        f"    logits: all[mean={logits_arr.mean():.3f} std={logits_arr.std():.3f} "
        + f"min={logits_arr.min():.3f} max={logits_arr.max():.3f}] "
        + f"pos_mean={pos_mean:.3f} unl_mean={unl_mean:.3f} "
        + f"separation={pos_mean - unl_mean:+.3f}",
        flush=True,
    )

    return evaluate_ranking(scores_arr, labels_arr, k=eval_k)


@dataclass(frozen=True, slots=True)
class EpochReport:
    epoch: int
    train_loss: float
    val_average_precision: float
    val_precision_at_k: float
    val_roc_auc: float
    is_best: bool


@dataclass(frozen=True, slots=True)
class FitResult:
    """Outcome of a training run.

    reports holds the per-epoch history. best_state_dict is a deep copy of the
    model weights at the epoch with the highest validation Proxy AUC (PAUC, the
    PU model-selection signal; deep-copied so later epochs do not mutate it), and
    best_epoch / best_val_pauc identify that epoch. After fit returns, the model
    has these best weights loaded, so it is ready to score or to save directly.
    """

    reports: list[EpochReport]
    best_state_dict: dict[str, Tensor]
    best_epoch: int
    best_val_pauc: float


def fit(
    model: Module,
    make_train_loader: Callable[[], Iterator[HeteroData]],
    make_val_loader: Callable[[], Iterator[HeteroData]],
    pu_label_of: Mapping[str, int],
    mapper_to_ids: Callable[[list[int]], list[str]],
    config: TrainConfig,
    optimizer: Optimizer | None = None,
    batch_transform: BatchTransform = _identity,
    train_batches: int | None = None,
    val_batches: int | None = None,
    device: torch.device | None = None,
    on_best: Callable[[dict[str, Tensor], int, float], None] | None = None,
) -> FitResult:
    """
    Train with early stopping on validation Proxy AUC (PAUC).

    make_train_loader / make_val_loader are zero-arg factories returning a fresh
    iterator of HeteroData batches for one epoch (fresh so each epoch reshuffles
    seeds and resamples neighborhoods). The train loader must use the training
    sampling regime (allow_val=False, allow_test=False); the val loader must use
    allow_val=True, allow_test=False. Returns one EpochReport per epoch run.

    train_batches / val_batches are the per-epoch batch counts, passed through to
    the progress bars for ETA (the loaders are lazy generators with no len).

    on_best, if given, is called the moment a new best epoch is found, with the
    best weights, epoch index, and val PAUC. Use it to persist the best
    checkpoint DURING training, so an interrupted long run still leaves the best
    model on disk rather than losing everything (the loop only returns at the
    very end).

    The optimizer defaults to AdamW(lr, weight_decay) over the model parameters;
    pass a pre-built optimizer to override.
    """
    dev = device if device is not None else select_device()
    _ = model.to(dev)
    opt: Optimizer = (
        optimizer
        if optimizer is not None
        else AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    )
    loss_fn = NonNegativePULoss(prior=config.prior)
    stopper = EarlyStopper(patience=config.patience, min_delta=config.min_delta)

    reports: list[EpochReport] = []
    best_state: dict[str, Tensor] = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_val_pauc = float("-inf")
    for epoch in range(config.max_epochs):
        train_loss = train_epoch(
            model,
            make_train_loader(),
            loss_fn,
            opt,
            pu_label_of,
            mapper_to_ids,
            batch_transform,
            total_batches=train_batches,
            epoch=epoch,
            device=dev,
        )
        val = validate(
            model,
            make_val_loader(),
            pu_label_of,
            mapper_to_ids,
            config.eval_k,
            total_batches=val_batches,
            epoch=epoch,
            device=dev,
        )
        is_best = stopper.update(val.roc_auc, epoch)
        if is_best:
            # deep-copy so subsequent epochs do not mutate the saved best weights
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_pauc = val.roc_auc
            # persist immediately, so an interrupted run still leaves the best
            # model on disk (the function only returns after all epochs)
            if on_best is not None:
                on_best(best_state, best_epoch, best_val_pauc)
        reports.append(
            EpochReport(
                epoch=epoch,
                train_loss=train_loss,
                val_average_precision=val.average_precision,
                val_precision_at_k=val.precision_at_k,
                val_roc_auc=val.roc_auc,
                is_best=is_best,
            )
        )

        # Print val metrics live, every epoch, so overfitting is visible as it
        # happens: watch val PAUC (the early-stop signal) against the falling
        # train loss. If train loss keeps dropping while PAUC stalls or falls,
        # the model is memorizing the few positives -- and the "since best"
        # countdown shows how close early stopping is to halting. AP and P@k are
        # diagnostics. (All metrics here use pu_label only.)
        marker = " *best" if is_best else f"  (no gain for {stopper.bad_epochs})"
        print(
            f"epoch {epoch:>2} | train_loss {train_loss:.4f} | "
            + f"val PAUC {val.roc_auc:.4f} | "
            + f"AP {val.average_precision:.4f} | "
            + f"P@{val.k} {val.precision_at_k:.4f} | "
            + f"pos {val.num_labeled_positives}/{val.num_evaluated}{marker}",
            flush=True,
        )

        if stopper.should_stop:
            print(
                f"early stop: val PAUC has not improved for {config.patience} epochs "
                + f"(best was epoch {best_epoch}, PAUC {best_val_pauc:.4f}).",
                flush=True,
            )
            break

    # leave the model holding its best weights, not the last (possibly worse) ones
    _ = model.load_state_dict(best_state)
    return FitResult(
        reports=reports,
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_val_pauc=best_val_pauc,
    )
