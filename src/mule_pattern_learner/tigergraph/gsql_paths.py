from pathlib import Path

_GSQL_RELPATHS: dict[str, str] = {
    # schema / loading
    "loading_job": "schema/loading_job.gsql",
    # features
    "account_account_degree": "features/account_account_degree.gsql",
    "derive_max_bins": "features/derive_max_bins.gsql",
    "fastrp": "features/fastrp.gsql",
    "identity_sharing": "features/identity_sharing.gsql",
    "money_flow": "features/money_flow.gsql",
    "pagerank": "features/pagerank.gsql",
    "temporal_features": "features/temporal_features.gsql",
    "triangle_clustering": "features/triangle_clustering.gsql",
    # community detection
    "weight_account_edges": "community_detection/weight_account_edges.gsql",
    "cluster_with_wcc": "community_detection/cluster_with_wcc.gsql",
    # entity resolution
    "match_parties": "entity_resolution/match_parties.gsql",
    "unify_parties": "entity_resolution/unify_parties.gsql",
    # sampling (ML runtime)
    "sample_khop_neighborhood": "sampling/sample_khop_neighborhood.gsql",
    "fetch_account_features": "sampling/fetch_account_features.gsql",
    "fetch_has_paid_features": "sampling/fetch_has_paid_features.gsql",
    # export
    "export_account_features": "export/export_account_features.gsql",
    "export_edges_by_type": "export/export_edges_by_type.gsql",
    "export_has_paid_edges": "export/export_has_paid_edges.gsql",
    # masking inputs
    "get_masking_inputs": "masking/get_masking_inputs.gsql",
    # experiments (throwaway probes / diagnostics)
    "diagnose_export": "experiments/diagnose_export.gsql",
    "diagnose_sentinels": "experiments/diagnose_sentinels.gsql",
    "probe_sample_clause": "experiments/probe_sample_clause.gsql",
    "get_split_accounts": "masking/get_split_accounts.gsql",
    "derive_reference_epoch": "features/derive_reference_epoch.gsql",
}


class GsqlPathError(KeyError):
    pass


def gsql_root() -> Path:
    """
    Absolute path to the repository's top-level gsql/ directory.
    """
    # this file: <repo>/src/mule_pattern_learner/tigergraph/gsql_paths.py
    # parents:    0 tigergraph  1 mule_pattern_learner  2 src  3 <repo>
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "gsql"


def gsql_path(query_name: str) -> Path:
    """
    Absolute path to the .gsql file defining the query.
    """
    relpath = _GSQL_RELPATHS.get(query_name)
    if relpath is None:
        known = ", ".join(sorted(_GSQL_RELPATHS))
        raise GsqlPathError(f"unknown query {query_name!r}; known: {known}")
    return gsql_root() / relpath


def query_names() -> list[str]:
    """All known installed-query names."""
    return sorted(_GSQL_RELPATHS)
