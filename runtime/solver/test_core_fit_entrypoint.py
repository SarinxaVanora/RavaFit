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

    def motion(production, cache, source):
        calls.append("motion")
        return {"registered": 1}

    def solve(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        assert cache.get("prepared") is True
        assert production.runtime_authority_ready is True
        calls.append("solve")
        return {}, {}, {}, {}

    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "prepare_cache", prepare)
    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "install_runtime_authority", install)
    monkeypatch.setattr(core_fit_entrypoint, "_install_motion_authority", motion)
    production = SimpleNamespace(_solve_garment_meshes=solve)

    core_fit_entrypoint.install_core_fit_entrypoint(production)
    result = production._solve_garment_meshes(object(), {}, set())

    assert calls == ["prepare", "install", "motion", "solve"]
    assert result[3]["core_fit_entrypoint"]["enabled"] is True
    assert result[3]["source_influence_capacity_registry"]["registered"] == 1


def test_strict_b14_entrypoint_cannot_bypass_core_authority(monkeypatch):
    calls = []

    def prepare_strict(production, cache):
        cache["strict_prepared"] = True
        calls.append("prepare-strict")
        return {"enabled": True, "strict_lane": True}

    def install(production):
        production.runtime_authority_ready = True
        calls.append("install")

    def motion(production, cache, source):
        calls.append("motion")
        return {"registered": 1}

    def direct(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        return {}, {}, {}, {}

    def strict(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        assert cache.get("strict_prepared") is True
        assert production.runtime_authority_ready is True
        calls.append("strict-solve")
        return {"garment": __import__("numpy").zeros((3, 3))}, {}, {}, {}

    def finalize(production, source, cache, positions):
        assert cache.get("strict_prepared") is True
        calls.append("strict-finalize")
        return positions, {"enabled": True, "converged": True}

    monkeypatch.setattr(core_fit_entrypoint.strict_fit_authority, "prepare_strict_cache", prepare_strict)
    monkeypatch.setattr(core_fit_entrypoint.strict_fit_authority, "finalize_strict_solution", finalize)
    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "install_runtime_authority", install)
    monkeypatch.setattr(core_fit_entrypoint, "_install_motion_authority", motion)

    production = SimpleNamespace(_solve_garment_meshes=direct, _solve_strict_b14_layers=strict)
    core_fit_entrypoint.install_core_fit_entrypoint(production)
    result = production._solve_strict_b14_layers(object(), {}, set())

    assert calls == ["prepare-strict", "install", "motion", "strict-solve", "strict-finalize"]
    assert result[3]["strict_core_fit_entrypoint"]["strict_lane"] is True
    assert result[3]["strict_final_authority"]["converged"] is True


def test_strict_b14_fails_closed_when_authority_cannot_prepare(monkeypatch):
    monkeypatch.setattr(core_fit_entrypoint.strict_fit_authority, "prepare_strict_cache", lambda production, cache: (_ for _ in ()).throw(ValueError("no macro authority")))
    monkeypatch.setattr(core_fit_entrypoint, "_install_motion_authority", lambda production, cache, source: {})
    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "install_runtime_authority", lambda production: None)

    def direct(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        return {}, {}, {}, {}

    def strict(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        raise AssertionError("strict solver must not run after authority preparation fails")

    production = SimpleNamespace(_solve_garment_meshes=direct, _solve_strict_b14_layers=strict)
    core_fit_entrypoint.install_core_fit_entrypoint(production)

    import pytest
    with pytest.raises(ValueError, match="no macro authority"):
        production._solve_strict_b14_layers(object(), {}, set())


def test_entrypoint_install_is_idempotent(monkeypatch):
    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "prepare_cache", lambda production, cache: {"enabled": True})
    monkeypatch.setattr(core_fit_entrypoint.core_fit_authority, "install_runtime_authority", lambda production: None)
    monkeypatch.setattr(core_fit_entrypoint, "_install_motion_authority", lambda production, cache, source: {})

    def solve(source, cache, body_mesh_names, mesh_filter=None, *args, **kwargs):
        return {}, {}, {}, {}

    production = SimpleNamespace(_solve_garment_meshes=solve)
    core_fit_entrypoint.install_core_fit_entrypoint(production)
    wrapped = production._solve_garment_meshes
    core_fit_entrypoint.install_core_fit_entrypoint(production)
    assert production._solve_garment_meshes is wrapped
