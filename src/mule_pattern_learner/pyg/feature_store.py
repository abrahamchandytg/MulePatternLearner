from typing import override

import torch
from torch import Tensor
from torch_geometric.data import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType, NodeType

from mule_pattern_learner.indexing.node_id_mapper import NodeIDMapper
from mule_pattern_learner.pyg.fetch import fetch_account_vertices
from mule_pattern_learner.features.nodes import (
    NUM_ACCOUNT_FEATURES,
    FeatureNormalizer,
    build_account_features,
)
from mule_pattern_learner.tigergraph.client import Client


class FeatureStoreError(RuntimeError):
    pass


_ACCOUNT: NodeType = "Account"
_NODE_FEATURE_ATTR: str = "x"

_FEATURE_DIMS: dict[NodeType, int] = {_ACCOUNT: NUM_ACCOUNT_FEATURES}


class TigerGraphFeatureStore(FeatureStore):
    """
    PyG FeatureStore backed by TigerGraph, sharing a backend's mapper.

    PyG calls _get_tensor with a TensorAttr carrying only (group_name,
    attr_name, index); index holds the global integer ids the sampler wrote
    into the node tensor (PyG's n_id), in batch-local order. This store reverses
    those integers to global string ids through the shared NodeIDMapper, fetches
    the rows from TigerGraph, applies the same transforms used everywhere else
    (node_features.py), and returns a tensor aligned row-for-row to index.

    The store is read-only: features live in TigerGraph, so put/remove are not
    supported (matching the remote-backend pattern, where the store is a view
    over the database rather than a writable cache).
    """

    _client: Client
    _mapper: NodeIDMapper
    _normalizer: FeatureNormalizer | None

    def __init__(
        self,
        client: Client,
        mapper: NodeIDMapper,
        normalizer: FeatureNormalizer | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._mapper = mapper
        self._normalizer = normalizer

    def _index_to_ids(self, group_name: NodeType, index: Tensor) -> list[str]:
        int_ids: list[int] = [int(i) for i in index.tolist()]
        return self._mapper.to_strings(group_name, int_ids)

    @override
    def _get_tensor(self, attr: TensorAttr) -> Tensor | None:
        group_name = attr.group_name
        attr_name = attr.attr_name
        index = attr.index
        if group_name is None or attr_name is None or index is None:
            raise FeatureStoreError(f"TensorAttr is not fully specified: {attr!r}")
        if attr_name != _NODE_FEATURE_ATTR:
            raise FeatureStoreError(
                f"unsupported attr_name {attr_name!r}; only {_NODE_FEATURE_ATTR!r}"
            )
        if group_name != _ACCOUNT:
            type_list = sorted(_FEATURE_DIMS)
            raise FeatureStoreError(
                f"no features for node type {group_name!r}; types are {type_list}"
            )
        if not isinstance(index, Tensor):
            raise FeatureStoreError(
                f"index must be a Tensor of node ids, got {type(index).__name__}"
            )

        string_ids = self._index_to_ids(group_name, index)
        vertices = fetch_account_vertices(self._client, string_ids)
        features = build_account_features(vertices)

        # Align the fetched rows to the requested index order: TigerGraph may
        # return vertices in any order, so reorder by the requested ids.
        aligned = self._align(features.node_ids, features.feats, string_ids)
        # Standardize with the training-fit stats (identical at train and eval).
        if self._normalizer is not None:
            aligned = self._normalizer.apply(aligned)
        return aligned

    def _align(self, fetched_ids: tuple[str, ...], feats: Tensor, requested: list[str]) -> Tensor:
        position: dict[str, int] = {sid: i for i, sid in enumerate(fetched_ids)}
        rows: list[int] = []
        for sid in requested:
            row = position.get(sid)
            if row is None:
                raise FeatureStoreError(f"feature row missing for id {sid!r}")
            rows.append(row)
        order = torch.tensor(rows, dtype=torch.long)
        return feats[order]

    @override
    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        _ = tensor
        _ = attr
        return False

    @override
    def _remove_tensor(self, attr: TensorAttr) -> bool:
        _ = attr
        return False

    @override
    def _get_tensor_size(self, attr: TensorAttr) -> tuple[int, ...] | None:
        group_name = attr.group_name
        if group_name is None:
            return None
        dim = _FEATURE_DIMS.get(group_name)
        if dim is None:
            return None
        return (dim,)

    @override
    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        return [
            TensorAttr(group_name=ntype, attr_name=_NODE_FEATURE_ATTR) for ntype in _FEATURE_DIMS
        ]
