from types import SimpleNamespace

import core_fit_entrypoint


def test_single_mesh_direct_entrypoint_prepares_core_fit_before_solver(monkeypatch):
    calls = []

    def prepare(production, cache):
        cache["prepared"] = True
        calls.append("prepare")
        return {"enabled": True, "test": "single-mesh-direct"}

    def install(production):
        production.runtime_authority_ready = True
        calls.append("install")

    def solve(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        assert cache.get("prepared") is True
        assert production.runtime_authority_ready is True
        calls.append("solve")
        return {}, {}, {}, {}

    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "prepare_cache", prepare)
    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "install_runtime_authority", install)
    production = SimpleNamespace(_solve_garment_meshes=solve)

    core_fit_entrypoint.install_core_fit_entrypoint(production)
    result = production._solve_garment_meshes(object(), {}, set())

    assert calls == ["prepare", "install", "solve"]
    assert result[3]["core_fit_entrypoint"]["enabled"] is True


def test_entrypoint_install_is_idempotent(monkeypatch):
    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "prepare_cache", lambda production, cache: {"enabled": True})
    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "install_runtime_authority", lambda production: None)

    def solve(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        return {}, {}, {}, {}

    production = SimpleNamespace(_solve_garment_meshes=solve)
    core_fit_entrypoint.install_core_fit_entrypoint(production)
    wrapped = production._solve_garment_meshes
    core_fit_entrypoint.install_core_fit_entrypoint(production)
    assert production._solve_garment_meshes is wrapped
