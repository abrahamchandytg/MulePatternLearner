import torch
from torch import Tensor
from torch_geometric.loader import NodeLoader

from mule_pattern_learner.features.nodes import FeatureNormalizer
from mule_pattern_learner.indexing.node_id_mapper import NodeIDMapper
from mule_pattern_learner.pyg.transform import HasPaidEdgeFeatureAttacher
from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.pyg.feature_store import TigerGraphFeatureStore
from mule_pattern_learner.pyg.graph_store import TigerGraphGraphStore
from mule_pattern_learner.pyg.sampler import TigerGraphHeteroSampler
from mule_pattern_learner.tigergraph.client import Client


class TigerGraphRemoteBackend:
    """Shared container wiring TigerGraph into PyG's remote-backend interfaces.

    Holds the single TigerGraph client and the one persistent NodeIDMapper that
    the sampler and the feature store must agree on: the sampler registers each
    sampled node's global string id and writes the assigned integer into PyG's
    node tensor, and the feature store later reverses those same integers back
    to string ids to fetch features. Because PyG's _get_tensor receives only
    (group_name, attr_name, index) and not the sampler's metadata, this mapping
    has to live on a shared object both sides reference (the FalkorDB pattern).

    The mapper persists across the run, so it grows with the set of distinct
    nodes actually sampled (bounded by the labeled seeds and their sampled
    neighborhoods), not the full graph. For true billion-scale with a fixed
    memory ceiling, the mapper could later evict stale batches (LRU); that is a
    future optimization and not required at the current scale.
    """

    _client: Client
    _mapper: NodeIDMapper

    def __init__(self, client: Client) -> None:
        self._client = client
        self._mapper = NodeIDMapper()

    @property
    def client(self) -> Client:
        """The shared TigerGraph client used for sampling and feature fetches."""
        return self._client

    @property
    def mapper(self) -> NodeIDMapper:
        """The shared, persistent global-id <-> integer mapper."""
        return self._mapper

    def make_sampler(
        self,
        seed_ids: tuple[str, ...],
        fanout: NeighborFanout | None = None,
        query_name: str = "sample_khop_neighborhood",
        allow_val: bool = True,
        allow_test: bool = True,
    ) -> TigerGraphHeteroSampler:
        """Build a k-hop sampler bound to this backend's client and mapper.

        The sampler shares this backend's mapper, so the integers it writes into
        PyG node tensors are reversible by the feature store built from the same
        backend.

        allow_val / allow_test control strict-inductive split filtering in the
        sampler query: when False, neighborhoods do not traverse into val / test
        accounts (so their features cannot leak into a seed's embedding). Train
        loaders pass both False; val loaders pass allow_val=True, allow_test=
        False; test loaders pass both True.
        """
        return TigerGraphHeteroSampler(
            client=self._client,
            seed_ids=seed_ids,
            mapper=self._mapper,
            fanout=fanout,
            query_name=query_name,
            allow_val=allow_val,
            allow_test=allow_test,
        )

    def make_feature_store(
        self, normalizer: FeatureNormalizer | None = None
    ) -> TigerGraphFeatureStore:
        """Build a feature store bound to this backend's client and mapper.

        The store shares this backend's mapper, so it reverses the same integer
        ids the sampler assigned back to global string ids when fetching
        features. This shared mapper is why the sampler and store must come from
        one backend (PyG's _get_tensor sees only the integers, not the sampler's
        metadata).

        normalizer, if given, standardizes account features as they are served
        (fit on the train split; see FeatureNormalizer).
        """
        return TigerGraphFeatureStore(
            client=self._client, mapper=self._mapper, normalizer=normalizer
        )

    def make_graph_store(self) -> TigerGraphGraphStore:
        """Build a graph store bound to this backend's client and mapper.

        The store shares this backend's mapper so any edge indices it exports
        use the same integer id space as the sampler and feature store. Note the
        sampler, not this store, drives batch sampling; the store serves whole
        edge types on demand (off the hot path).
        """
        return TigerGraphGraphStore(client=self._client, mapper=self._mapper)

    def remote_backend(
        self,
    ) -> tuple[TigerGraphFeatureStore, TigerGraphGraphStore]:
        """Return the (feature_store, graph_store) pair PyG's NodeLoader wants.

        Both share this backend's client and mapper, so the data=(feature_store,
        graph_store) tuple passed to a loader is internally consistent.
        """
        return (self.make_feature_store(), self.make_graph_store())

    def make_loader(
        self,
        seed_ids: tuple[str, ...],
        reference_epoch_s: float,
        max_bins: int,
        fanout: NeighborFanout | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        allow_val: bool = True,
        allow_test: bool = True,
        normalizer: FeatureNormalizer | None = None,
    ) -> NodeLoader:
        """Build a PyG NodeLoader that yields HeteroData batches from TigerGraph.

        Wires the (feature_store, graph_store) pair, the k-hop sampler, and the
        HAS_PAID edge-feature transform, all sharing this backend's client and
        mapper. input_nodes is the Account seed set as integer indices into
        seed_ids; the sampler maps those to global ids before sampling.

        Node features arrive via the feature store; HAS_PAID edge features are
        attached by the transform, since PyG's remote feature store serves node
        features only.

        allow_val / allow_test enforce strict-inductive sampling (see
        make_sampler). For leakage-free training pass allow_val=False,
        allow_test=False; for validation pass allow_val=True, allow_test=False;
        for test pass both True.
        """
        feature_store = self.make_feature_store(normalizer=normalizer)
        graph_store = self.make_graph_store()
        sampler = self.make_sampler(
            seed_ids=seed_ids,
            fanout=fanout,
            allow_val=allow_val,
            allow_test=allow_test,
        )
        transform = HasPaidEdgeFeatureAttacher(
            client=self._client,
            mapper=self._mapper,
            reference_epoch_s=reference_epoch_s,
            max_bins=max_bins,
        )
        seed_index: Tensor = torch.arange(len(seed_ids), dtype=torch.long)
        return NodeLoader(
            data=(feature_store, graph_store),
            node_sampler=sampler,
            input_nodes=("Account", seed_index),
            transform=transform,
            batch_size=batch_size,
            shuffle=shuffle,
        )
