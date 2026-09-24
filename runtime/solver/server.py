from __future__ import annotations

import sys
from pathlib import Path

# SolverHost is launched with Python -I in production. Isolated mode deliberately
# does not prepend the script directory to sys.path, so establish our own trusted
# local module root before importing sibling runtime modules.
SOLVER_TOOLS = Path(__file__).resolve().parent
_solver_tools = str(SOLVER_TOOLS)
if _solver_tools not in sys.path:
    sys.path.insert(0, _solver_tools)

import server_core as _server
from core_fit_entrypoint import install_core_fit_entrypoint


_original_load_production_module = _server._load_production_module


def _load_production_module():
    production = _original_load_production_module()
    install_core_fit_entrypoint(production)
    return production


_server._load_production_module = _load_production_module


if __name__ == "__main__":
    if "--health-json" in sys.argv[1:]:
        raise SystemExit(_server._health_cli())
    _server.main()
