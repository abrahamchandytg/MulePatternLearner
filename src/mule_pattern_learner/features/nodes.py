import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import cast

import torch
from torch import Tensor

from mule_pattern_learner.schema.specs import ACCOUNT_NUMERIC_FEATURES


class NodeFeatureError(ValueError):
    pass


class Transform(Enum):
    """How a raw scalar feature is mapped before entering the model."""

    LOG1P = auto()
    SYMLOG = auto()
    IDENTITY = auto()
    BOOLEAN = auto()
    CLAMP_SENTINEL = auto()


_MISSING_SENTINEL: float = -1.0


def log1p_compress(value: float) -> float:
    """Compress a heavy-tailed non-negative quantity via log(1 + x)."""
    return math.log1p(max(value, 0.0))


def symlog_compress(value: float) -> float:
    """Sign-preserving log compression for signed heavy-tailed values.

    Computes sign(x) * log(1 + |x|): continuous through zero (0 -> 0), keeps
    the sign, and reduces to log1p_compress for non-negative inputs. Preferred
    over sign(x) * log(|x|), which is undefined at zero and amplifies near-zero
    noise.
    """
    sign = 1.0 if value >= 0.0 else -1.0
    return sign * math.log1p(abs(value))


_ACCOUNT_FEATURE_TRANSFORMS: dict[str, Transform] = {
    "pagerank": Transform.IDENTITY,
    "com_size": Transform.LOG1P,
    "aa_degree": Transform.LOG1P,
    "triangle_count": Transform.LOG1P,
    "clustering_coef": Transform.IDENTITY,
    "in_degree": Transform.LOG1P,
    "out_degree": Transform.LOG1P,
    "in_amount": Transform.LOG1P,
    "out_amount": Transform.LOG1P,
    "in_txn_count": Transform.LOG1P,
    "out_txn_count": Transform.LOG1P,
    "fan_in_ratio": Transform.IDENTITY,
    "fan_out_ratio": Transform.IDENTITY,
    "pass_through_ratio": Transform.IDENTITY,
    "net_flow": Transform.SYMLOG,
    "activity_span_days": Transform.IDENTITY,
    "days_since_last_txn": Transform.CLAMP_SENTINEL,
    "account_age_days": Transform.CLAMP_SENTINEL,
    "mean_inter_txn_days": Transform.CLAMP_SENTINEL,
    "txn_per_active_day": Transform.IDENTITY,
    "burst_ratio": Transform.IDENTITY,
    "active_bin_count": Transform.LOG1P,
    "activity_concentration": Transform.IDENTITY,
    "peak_bin_fraction": Transform.IDENTITY,
    "early_late_ratio": Transform.SYMLOG,
    "in_out_lag_days": Transform.SYMLOG,
    "device_share_cnt": Transform.LOG1P,
    "ip_share_cnt": Transform.LOG1P,
    "phone_share_cnt": Transform.LOG1P,
    "email_share_cnt": Transform.LOG1P,
    "is_external": Transform.BOOLEAN,
}


def account_feature_names() -> tuple[str, ...]:
    """Account feature names in tensor-column order (matches schema spec)."""
    return ACCOUNT_NUMERIC_FEATURES


def _validate_transform_coverage() -> None:
    spec = set(ACCOUNT_NUMERIC_FEATURES)
    mapped = set(_ACCOUNT_FEATURE_TRANSFORMS)
    missing = spec - mapped
    extra = mapped - spec
    if missing or extra:
        raise NodeFeatureError(
            "account transform policy out of sync with ACCOUNT_NUMERIC_FEATURES: "
            + f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )


_validate_transform_coverage()

NUM_ACCOUNT_FEATURES: int = len(ACCOUNT_NUMERIC_FEATURES)


@dataclass(frozen=True, slots=True)
class NodeFeatures:
    node_ids: tuple[str, ...]
    feats: Tensor
    feature_names: tuple[str, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    def __post_init__(self) -> None:
        n = len(self.node_ids)
        f = len(self.feature_names)
        if tuple(self.feats.shape) != (n, f):
            raise NodeFeatureError(f"feats shape {tuple(self.feats.shape)} != ({n}, {f})")


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise NodeFeatureError(f"{field}: expected number, got {type(value).__name__}")


def _apply_transform(raw: float, transform: Transform) -> float:
    if transform is Transform.LOG1P:
        return log1p_compress(raw)
    if transform is Transform.SYMLOG:
        return symlog_compress(raw)
    if transform is Transform.BOOLEAN:
        return 1.0 if raw != 0.0 else 0.0
    if transform is Transform.CLAMP_SENTINEL:
        return 0.0 if raw == _MISSING_SENTINEL else raw
    return raw


def _vertex_attributes(vertex: object) -> tuple[str, dict[str, object]]:
    if not isinstance(vertex, dict):
        raise NodeFeatureError(f"vertex is not a dict: {vertex!r}")
    vertex_typed: dict[str, object] = {
        str(k): v for k, v in cast(dict[object, object], vertex).items()
    }
    node_id = vertex_typed.get("v_id")
    if not isinstance(node_id, str) or not node_id:
        raise NodeFeatureError(f"vertex missing 'v_id': {vertex_typed!r}")
    attrs = vertex_typed.get("attributes")
    if not isinstance(attrs, dict):
        raise NodeFeatureError(f"vertex missing attributes dict: {vertex_typed!r}")
    attrs_typed: dict[str, object] = {
        str(k): v for k, v in cast(dict[object, object], attrs).items()
    }
    return node_id, attrs_typed


def _build_account_row(attrs: dict[str, object]) -> list[float]:
    row: list[float] = []
    for name in ACCOUNT_NUMERIC_FEATURES:
        raw = _as_float(attrs.get(name, 0.0), name)
        row.append(_apply_transform(raw, _ACCOUNT_FEATURE_TRANSFORMS[name]))
    return row


def build_account_features(vertices: Sequence[object]) -> NodeFeatures:
    node_ids: list[str] = []
    rows: list[float] = []

    for vertex in vertices:
        node_id, attrs = _vertex_attributes(vertex)
        node_ids.append(node_id)
        rows.extend(_build_account_row(attrs))

    n = len(node_ids)
    feats = torch.tensor(rows, dtype=torch.float32).reshape(n, NUM_ACCOUNT_FEATURES)

    return NodeFeatures(
        node_ids=tuple(node_ids),
        feats=feats,
        feature_names=ACCOUNT_NUMERIC_FEATURES,
    )


@dataclass(frozen=True, slots=True)
class FeatureNormalizer:
    """Per-feature z-score applied AFTER the log/symlog transforms above.

    The log family tames heavy tails but leaves features on different centers and
    spreads (a log-amount near 14 sitting beside a ratio in [0, 1] and a pagerank
    near 1e-3). Standardizing to zero mean / unit variance puts them on one scale
    so no single feature dominates the first linear layer and drives activations
    (and logits) to extremes.

    mean and std are length-NUM_ACCOUNT_FEATURES, in ACCOUNT_NUMERIC_FEATURES
    column order, and MUST be fit on the training split only (see fit_normalizer)
    then reused unchanged for val / test / inference -- otherwise val and test
    feature statistics leak into training. They are saved with the checkpoint, so
    scoring reuses the exact training-time standardization.
    """

    mean: Tensor
    std: Tensor

    def __post_init__(self) -> None:
        if self.mean.shape != (NUM_ACCOUNT_FEATURES,) or self.std.shape != (NUM_ACCOUNT_FEATURES,):
            raise NodeFeatureError(
                f"normalizer mean/std must be ({NUM_ACCOUNT_FEATURES},); "
                + f"got {tuple(self.mean.shape)} / {tuple(self.std.shape)}"
            )

    def apply(self, feats: Tensor) -> Tensor:
        # std is floored away from zero in fit_normalizer, so this never divides
        # by zero (a constant feature becomes all-zeros, which is harmless).
        return (feats - self.mean.to(feats.device)) / self.std.to(feats.device)


def normalizer_from_features(feats: Tensor) -> FeatureNormalizer:
    """Fit a FeatureNormalizer from an (n, NUM_ACCOUNT_FEATURES) feature matrix.

    Caller is responsible for passing TRAIN-split features only. A small floor on
    std keeps constant / near-constant columns from producing huge or NaN values.
    """
    if feats.ndim != 2 or feats.shape[1] != NUM_ACCOUNT_FEATURES:
        raise NodeFeatureError(
            f"expected (n, {NUM_ACCOUNT_FEATURES}) features; got {tuple(feats.shape)}"
        )
    mean = feats.mean(dim=0)
    std = feats.std(dim=0, unbiased=False).clamp_min(1e-6)
    return FeatureNormalizer(mean=mean, std=std)
