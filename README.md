# Mule Pattern Learner

Learns mule and money-laundering patterns from transactional data by training a graph neural network (GNN) directly from TigerGraph.

The project contains:

1. **A GSQL layer** on TigerGraph that resolves entities, detects WCC communities, and derives node/edge features on the graph.
2. **A Python (PyTorch Geometric) layer** that uses TigerGraph as a remote backend, samples neighbourhoods, and trains a node classifier that ranks accounts by how mule-like they are.

For this project, you would need to be connected to a working instance of
TigerGraph. The target graph name used in this project is `Mule_Pattern_Learner`.


## Setup

```bash
pip install -e ".[model]"        # add ,baseline,dev as needed, or use [all]
```

Create a `.env` for the TigerGraph connection (read by `Settings`):

```
HOST=https://your-tg-host
GRAPHNAME=Mule_Pattern_Learner
SECRET=your_restpp_secret
```

## How the queries are organised

Every installed query lives under `gsql/` and is registered in `src/mule_pattern_learner/tigergraph/gsql_paths.py`, which maps a short **registry name** to its `.gsql` file. A registry name is not always the installed query name; the ones that matter most differ:

| Registry name | Installed query | Group |
| --- | --- | --- |
| `match_parties` | `match_parties` | entity resolution |
| `unify_parties` | `unify_parties` | entity resolution |
| `weight_account_edges` | `account_account_with_weights` | community detection |
| `cluster_with_wcc` | `tg_wcc_account_with_weights` | community detection |
| `pagerank` | `tg_pagerank_wt_account` | features |
| `fastrp` | `tg_fastRP` | features |
| `money_flow` | `account_money_flow_features` | features |
| `temporal_features` | `account_temporal_features` | features |
| `triangle_clustering` | `account_triangle_clustering` | features |
| `identity_sharing` | `account_identity_sharing_features` | features |
| `account_account_degree` | `account_aa_degree_feature` | features |
| `derive_reference_epoch` | `derive_reference_epoch` | features (derivation) |
| `derive_max_bins` | `derive_max_bins` | features (derivation) |
| `get_split_accounts` | `get_split_accounts` | masking |
| `get_masking_inputs` | `get_masking_inputs` | masking |
| `sample_khop_neighborhood` | `sample_khop_neighborhood` | sampling (runtime) |
| `fetch_account_features` | `fetch_account_features` | sampling (runtime) |
| `fetch_has_paid_features` | `fetch_has_paid_features` | sampling (runtime) |
| `export_account_features` / `export_edges_by_type` / `export_has_paid_edges` | same | export |

### Installing queries

`src/mule_pattern_learner/tigergraph/gsql_install.py` installs a query by reading its file in this repository, 
extracting the `CREATE [OR REPLACE] [DISTRIBUTED] QUERY <name>` name, and installing it onto the graph:

```python
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.tigergraph.gsql_install import install_query

client = Client(Settings())
install_query(client, "match_parties", drop_first=True)
# ...install the rest before running them.
```

Install a query before you run it.

## Query execution order

The graph must be prepared in stages, because later queries read structure that earlier queries write. Run them in this order.

### 0. Schema and load

- `gsql/schema/schema.gsql` defines the vertices and edges (accounts, parties, PII values, transactions, and the derived types `Resolved_Entity`, `Connected_Component`, and so on).
- `gsql/schema/loading_job.gsql` (registry `loading_job`) loads the transaction/account/PII data into that schema. Files were loaded into the GUI in GraphStudio, adjust accordingly.

`scripts/pipeline/pre_graph/temporal_flow_aggs.py` precomputes temporal flow aggregates before/with loading.

### 1. Entity resolution (order is mandatory)

Two queries, run strictly in this order:

1. **`match_parties`** scans every pair of `Party` vertices that share a PII value (email, phone, birthdate, address parts, or name/address/street MinHash buckets), each reachable through the `Has_*` edges. It accumulates a weighted match score per shared attribute, skips PII values whose degree is implausibly high (the `pii_*_connections_limit` parameters, so that a shared city does not link thousands of people), and writes a **`Same_As`** edge between any two parties whose combined score clears `threshold`.
2. **`unify_parties`** runs a weakly-connected-components pass over those `Same_As` edges (iterative minimum-component-id propagation), then materialises one **`Resolved_Entity`** vertex per component and a **`Party_In_Entity`** edge from each party to its entity.

`unify_parties` reads the `Same_As` edges that `match_parties` produces, so running it first, or without `match_parties`, yields no resolved entities.

### 2. Community detection / WCC (order is mandatory)

Two queries, run strictly in this order:

1. **`account_account_with_weights`** (file `weight_account_edges.gsql`) deletes any existing `Account_Account` edges, then walks the `HAS_PAID` transaction edges and re-inserts a symmetric `Account_Account` edge between each pair of accounts, weighted by the total number of transactions between them (`min_edge_weight` drops trivial links).
2. **`tg_wcc_account_with_weights`** (file `cluster_with_wcc.gsql`) runs weakly-connected components over those `Account_Account` edges, keeping only edges with `weight >= min_link_weight`. It writes `com_id` / `com_size` back onto each `Account` and materialises `Connected_Component` vertices with `Account_In_Ring` edges.

The WCC query consumes the weighted `Account_Account` edges the first query builds, so the weighting step has to run first.

### 3. Feature derivation

With entities resolved and communities detected, derive the node and edge features. These are largely independent of one another and can run in any order, with a few dependencies:

- `account_temporal_features` (registry `temporal_features`) builds per-account temporal transaction features (binned amounts/counts). Run it **before** `derive_max_bins`, which reads the resulting `num_bins`.
- `account_identity_sharing_features` (registry `identity_sharing`) derives features from shared identity, so it expects entity resolution (stage 1) to have run.
- `account_aa_degree_feature` (registry `account_account_degree`) reads the weighted `Account_Account` graph from stage 2.
- `tg_pagerank_wt_account` (weighted PageRank), `tg_fastRP` (FastRP structural embeddings), `account_money_flow_features` (in/out money-flow aggregates), and `account_triangle_clustering` (triangle/clustering features) can run any time after stages 1 and 2.
- `derive_reference_epoch` returns the latest transaction time in the graph, the snapshot epoch that recency features are measured against.

`derive_max_bins` and `derive_reference_epoch` are also invoked at training time by `src/mule_pattern_learner/tigergraph/derivation.py`, which folds them into a single `GraphTemporalSpec` so the loader's edge-feature padding and the model's `edge_dim` cannot disagree.

### 4. Masking and splits

- `get_split_accounts` assigns accounts to train / validation / test.
- `get_masking_inputs` produces the inputs for positive-unlabelled (PU) masking (revealed vs hidden positives).

`scripts/pipeline/after_load/masking.py` drives this stage; the Python side lives in `src/mule_pattern_learner/data/splitting.py` and `pu_masking.py`.

### 5. Runtime queries (used during training, not a one-off step)

Installed once and called repeatedly by the PyG remote backend while training:

- `sample_khop_neighborhood` samples a k-hop neighbourhood around a batch of seed accounts.
- `fetch_account_features` and `fetch_has_paid_features` fetch node and edge features for the sampled subgraph.

The `export_*` queries are an alternative to runtime sampling: they bulk-export account features and edges for fully offline training.

## Scripts

- **Install queries:** `tigergraph/gsql_install.py` (`install_query`), with the file registry in `gsql_paths.py`.
- **Pre-graph:** `scripts/pipeline/pre_graph/temporal_flow_aggs.py` (temporal flow aggregates before load).
- **After load:** `scripts/pipeline/after_load/masking.py` (splits and PU masks).
- **Train:** `mule-train` (entry point `mule_pattern_learner.training.train:main`). The training loop (`training/loop.py`) drives a PyG model (`pyg/model.py`) over a TigerGraph-backed remote dataset (`pyg/backend.py`, `graph_store.py`, `feature_store.py`, `sampler.py`), using `derive_temporal_spec` for edge widths.
- **Evaluate:** `scripts/experiments/evaluate_hidden.py` (recall on hidden, never-revealed positives, the generalization headline) and `scripts/experiments/check_val_mules.py`.
- **Diagnostics and demos:** `scripts/experiments/diagnose*.py`, `sample_probe.py`, `khop_probe.py`, and runnable walk-throughs under `scripts/demos/` (backend, sampler, feature store, loader, model, and so on). Not required to run, but helps visualize data. 

End-to-end flow:

```
install queries
  -> load data (+ pre_graph aggregates)
  -> match_parties -> unify_parties                               # entity resolution
  -> account_account_with_weights -> tg_wcc_account_with_weights  # WCC
  -> feature queries (pagerank, fastrp, money_flow, temporal, triangle, identity, aa-degree)
  -> get_split_accounts -> get_masking_inputs                     # splits + PU masks
  -> mule-train                                                   # GNN training (runtime sampling)
  -> evaluate_hidden                                              # generalization check
```
## Notes

- Connection settings come from `.env` via `Settings` (`host`, `graphname`, `secret`); all three are required.
- Runs are seeded for reproducibility (`training/seeds.py`).
- Several queries carry a raised `query_timeout`, since entity resolution and WCC scan the full graph.

## License

MIT (author: Abraham Chandy).
