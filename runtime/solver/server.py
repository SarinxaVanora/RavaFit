from __future__ import annotations

import sys

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
