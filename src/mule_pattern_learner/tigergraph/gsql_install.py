"""
Install GSQL queries onto the graph from files in this repo.
"""

import re
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.gsql_paths import gsql_path

_FAILURE_MARKERS: tuple[str, ...] = (
    "error",
    "fail",
    "could not",
    "cannot",
    "not valid",
    "syntax",
)

# CREATE [OR REPLACE] [DISTRIBUTED] QUERY <name> -- captures the installed name.
_CREATE_QUERY_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:DISTRIBUTED\s+)?QUERY\s+([A-Za-z_]\w*)"
)


class GsqlInstallError(RuntimeError):
    pass


def get_query_from_file(registry_name: str) -> str:
    """
     Query name for a gsql_paths registry key.

    Parses `CREATE [OR REPLACE] [DISTRIBUTED] QUERY <name>` from the .gsql file.
    """
    text = gsql_path(registry_name).read_text(encoding="utf-8")
    match = _CREATE_QUERY_RE.search(text)
    if match is None:
        raise GsqlInstallError(
            f"{registry_name}: no CREATE QUERY found in source; not an installable query "
            "(a loading job or schema script installs differently)."
        )
    return match.group(1)


def _run_gsql(client: Client, statement: str) -> str:
    result = client.conn.gsql(statement)
    if not isinstance(result, str):
        raise GsqlInstallError(f"expected text from gsql(), got {type(result).__name__}")
    return result


def _check_output(registry_name: str, action: str, output: str) -> None:
    low = output.lower()
    if any(marker in low for marker in _FAILURE_MARKERS):
        raise GsqlInstallError(
            f"{action} {registry_name!r} reported a problem:\n{output.strip()[:800]}"
        )


def install_query(client: Client, registry_name: str, drop_first: bool = True) -> str:
    """
    Install one query from its .gsql file; return the install log.

    """
    path = gsql_path(registry_name)
    if not path.is_file():
        raise GsqlInstallError(f"{registry_name}: source file not found at {path}")
    name = get_query_from_file(registry_name)
    text = path.read_text(encoding="utf-8")

    if drop_first:
        # a failed drop is tolerated only when the query was simply not installed
        drop_out = _run_gsql(client, f"USE GRAPH {client.graphname}\nDROP QUERY {name}\n")
        low = drop_out.lower()
        if any(m in low for m in _FAILURE_MARKERS) and "not exist" not in low:
            raise GsqlInstallError(f"drop {name!r} failed:\n{drop_out.strip()[:800]}")

    install_out = _run_gsql(client, text + f"\nINSTALL QUERY {name}\n")
    _check_output(registry_name, "install", install_out)
    return install_out


def install_queries(
    client: Client, registry_names: list[str], drop_first: bool = True
) -> dict[str, str]:
    """
    Installs queries in order; return {registry_name: install log}.
    """
    logs: dict[str, str] = {}
    for registry_name in registry_names:
        logs[registry_name] = install_query(client, registry_name, drop_first=drop_first)
    return logs


def run_query(
    client: Client, registry_name: str, params: dict[str, object] | None = None
) -> list[object]:
    """
    Run an installed query by its gsql_paths registry key.
    """
    name = get_query_from_file(registry_name)
    return cast(
        list[object],
        client.conn.runInstalledQuery(name, params if params is not None else {}),
    )
