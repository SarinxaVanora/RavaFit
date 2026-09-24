from __future__ import annotations
import importlib.util
import hashlib
import json
import math
import numbers
import os
import re
import site
import sys
import traceback
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
SOLVER_TOOLS = Path(__file__).resolve().parent
RBODY_TOOLS = RUNTIME_ROOT / "rbody"
B14_FROZEN = RUNTIME_ROOT / "b14_frozen"
B14_SCRIPTS = B14_FROZEN / "scripts"
PACKAGES = RUNTIME_ROOT / "packages"

# Third-party wheels live under Runtime/packages; register that before scientific imports.
if PACKAGES.exists():
    site.addsitedir(str(PACKAGES))
    value = str(PACKAGES)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

for p in reversed((SOLVER_TOOLS, RBODY_TOOLS, B14_SCRIPTS)):
    value = str(p)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

_PRODUCTION_MODULE_NAME = "_ravafit_production_b14"

def _load_production_module():
    expected = (SOLVER_TOOLS / "production_b14.py").resolve()
    if not expected.exists():
        raise FileNotFoundError(expected)

    existing = sys.modules.get(_PRODUCTION_MODULE_NAME)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == expected:
        return existing

    spec = importlib.util.spec_from_file_location(_PRODUCTION_MODULE_NAME, expected)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RavaFit production solver from {expected}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PRODUCTION_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PRODUCTION_MODULE_NAME, None)
        raise
    return module


def _json_safe(value):
    """Return a strict JSON-safe SolverHost payload."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # Make NumPy-ish diagnostics JSON-safe without importing NumPy at module import time.
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _json_safe(to_list())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return value


def _response(req_id, **payload):
    out = _json_safe({"id": req_id, **payload})
    # Keep allow_nan=False so bad protocol values fail here rather than reaching .NET.
    sys.stdout.write(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def _health():
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "trimesh", "torch", "numba", "llvmlite"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except BaseException as ex:
            versions[name] = f"missing: {type(ex).__name__}: {ex}"

    conversion_error = None
    conversion_module = None
    production_revision = None
    conversion_ready = False
    native_body_graft_ready = False
    coverage_ready = False
    source_body_detection_ready = False
    inspect_parts_ready = False
    hide_parts_ready = False
    tag_parts_ready = False
    split_parts_ready = False
    piercing_ready = False

    try:
        production = _load_production_module()
        conversion_module = str(Path(production.__file__).resolve())
        production_revision = getattr(production, "PRODUCTION_REVISION", None)
        if hasattr(production, "runtime_self_test"):
            production.runtime_self_test()
        conversion_ready = callable(getattr(production, "convert", None))
        if not conversion_ready:
            conversion_error = "production_b14.convert is not callable"
    except BaseException as ex:
        conversion_error = f"{type(ex).__name__}: {ex}"

    try:
        from native_body_graft import graft_native_body as native_graft
        native_body_graft_ready = callable(native_graft)
    except BaseException as ex:
        if conversion_error is None:
            conversion_error = f"native body graft unavailable: {type(ex).__name__}: {ex}"

    try:
        from coverage_analysis import analyse_coverage, detect_source_bodies
        coverage_ready = callable(analyse_coverage)
        source_body_detection_ready = callable(detect_source_bodies)
    except BaseException as ex:
        if conversion_error is None:
            conversion_error = f"coverage analysis unavailable: {type(ex).__name__}: {ex}"

    try:
        from customise_mod import extract_piercings_glb, hide_mdl_parts, inspect_mdl_parts, replace_piercings_glb, split_mdl_parts, tag_mdl_parts
        inspect_parts_ready = callable(inspect_mdl_parts)
        hide_parts_ready = callable(hide_mdl_parts)
        tag_parts_ready = callable(tag_mdl_parts)
        split_parts_ready = callable(split_mdl_parts)
        piercing_ready = callable(extract_piercings_glb) and callable(replace_piercings_glb)
    except BaseException as ex:
        if conversion_error is None:
            conversion_error = f"customisation tools unavailable: {type(ex).__name__}: {ex}"

    return {
        "ok": True,
        "runtime_root": str(RUNTIME_ROOT),
        "versions": versions,
        "frozen_b14_present": (B14_FROZEN / "scripts" / "ffxiv_lobofit.py").exists(),
        "conversion_module": conversion_module,
        "production_revision": production_revision,
        "conversion_error": conversion_error,
        "capabilities": {
            "catalogue": True,
            "import_body": True,
            "prepare_cache": True,
            "extract_payload": True,
            "convert": conversion_ready,
            "graft_native_body": native_body_graft_ready,
            "analyze_coverage": coverage_ready,
            "detect_source_bodies": source_body_detection_ready,
            "inspect_mdl_parts": inspect_parts_ready,
            "hide_mdl_parts": hide_parts_ready,
            "tag_mdl_parts": tag_parts_ready,
            "split_mdl_parts": split_parts_ready,
            "piercing_customisation": piercing_ready,
        },
    }


def _catalogue(payload):
    directory = Path(payload["directory"]).expanduser().resolve()
    result = []
    if not directory.exists():
        return {"ok": True, "libraries": result}
    for path in sorted(directory.glob("*.rbody")):
        try:
            with zipfile.ZipFile(path) as z:
                manifest = json.loads(z.read("manifest.json"))
                catalogue = json.loads(z.read("catalogue.json"))
            result.append({
                "path": str(path),
                "manifest": manifest,
                "catalogue": catalogue,
            })
        except Exception as ex:
            result.append({"path": str(path), "error": str(ex)})
    return {"ok": True, "libraries": result}


def _slug(value):
    import re
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").casefold()).strip("-")
    return text or "body"


def _analysis_geometry_id(surface):
    h = hashlib.sha256()
    h.update(surface["V"].astype("<f4", copy=False).tobytes())
    h.update(surface["F"].astype("<i4", copy=False).tobytes())
    return h.hexdigest()


def _sex_for_race_code(race_code):
    try:
        prefix = int(str(race_code)[:2])
    except Exception:
        return None
    if 1 <= prefix <= 18:
        return "male" if prefix % 2 else "female"
    return None


def _preferred_body_model(rows):
    """Resolve duplicate redirects without guessing between two real body shapes."""
    by_hash = {}
    for row in rows:
        raw = Path(row["physical_path"]).resolve().read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        by_hash.setdefault(digest, []).append((row, raw))
    if len(by_hash) == 1:
        return next(iter(by_hash.values()))[0]

    def rank(item):
        path = str(item[0]["game_path"]).replace("\\", "/").casefold()
        if "/obj/body/" in path:
            return 0
        if "/equipment/e0000/" in path:
            return 1
        return 2

    candidates = [values[0] for values in by_hash.values()]
    best_rank = min(rank(item) for item in candidates)
    best = [item for item in candidates if rank(item) == best_rank]
    if len(best) == 1:
        return best[0]
    paths = ", ".join(str(item[0]["game_path"]) for item in best)
    raise ValueError(f"The selected Penumbra option contains multiple different body models for the same slot/race ({paths}). Choose a more specific body option.")


def _import_body(payload):
    from rbody_v3_core import body_surface_view, hash_solver_identity, parse_rigged_mdl
    from rbody_v3_loader import close_cached_rbodies

    output = Path(payload["output"]).expanduser().resolve()
    collection = str(payload.get("collection") or "Custom").strip() or "Custom"
    body_name = str(payload.get("body_name") or "").strip()
    variant_name = str(payload.get("variant_name") or "").strip()
    source_mod = str(payload.get("source_mod") or "Custom body").strip() or "Custom body"
    source_group = str(payload.get("source_group") or "").strip()
    source_option = str(payload.get("source_option") or "").strip()
    models = list(payload.get("models") or [])
    if not body_name:
        raise ValueError("Body name is required.")
    if not variant_name:
        raise ValueError("Variant name is required.")
    if not models:
        raise ValueError("No body models were supplied.")

    allowed_slots = {"Chest", "Legs", "Hands", "Feet"}
    grouped = {}
    for row in models:
        slot = str(row.get("slot") or "")
        race_code = str(row.get("race_code") or "")
        game_path = str(row.get("game_path") or "").replace("\\", "/")
        physical_path = str(row.get("physical_path") or "")
        if slot not in allowed_slots:
            raise ValueError(f"Unsupported body slot: {slot or '<missing>'}")
        if len(race_code) != 4 or not race_code.isdigit():
            raise ValueError(f"Body model has no usable c#### race code: {game_path}")
        physical = Path(physical_path).resolve()
        if not physical.is_file():
            raise FileNotFoundError(physical)
        grouped.setdefault((slot, race_code), []).append({
            "slot": slot,
            "race_code": race_code,
            "game_path": game_path,
            "relative_path": str(row.get("relative_path") or Path(physical_path).name).replace("\\", "/"),
            "physical_path": str(physical),
        })

    selected = []
    for (slot, race_code), rows in sorted(grouped.items()):
        row, raw = _preferred_body_model(rows)
        ref = parse_rigged_mdl(raw)
        if not ref.get("solver_compatible"):
            warnings = "; ".join(str(x) for x in (ref.get("warnings") or [])[:3])
            raise ValueError(f"{slot} {race_code} is not solver-compatible{': ' + warnings if warnings else ''}.")
        surface = body_surface_view(ref)
        payload_id = hashlib.sha256(raw).hexdigest()
        analysis_id = _analysis_geometry_id(surface)
        selected.append((row, raw, ref, surface, payload_id, analysis_id))

    close_cached_rbodies()
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {"format": "RBODY", "version": 4, "collection": collection}
    catalogue = {
        "format": "RBODY", "version": 4, "collection": collection,
        "slot_order": ["Chest", "Legs", "Hands", "Feet"], "bodies": [],
        "payload_semantics": "Immutable pristine source MDLs imported by RavaFit.",
    }
    payload_index = {}
    old_archive = None
    if output.exists():
        old_archive = zipfile.ZipFile(output, "r")
        try:
            old_manifest = json.loads(old_archive.read("manifest.json"))
            if old_manifest.get("format") != "RBODY" or int(old_manifest.get("version", 0)) < 3:
                raise ValueError(f"{output.name} is not an RBODY V3+ catalogue.")
            catalogue = json.loads(old_archive.read("catalogue.json"))
            payload_index = json.loads(old_archive.read("payload_index.json"))
            catalogue["format"] = "RBODY"
            catalogue["version"] = 4
            catalogue["collection"] = collection
        except Exception:
            old_archive.close()
            raise

    body_id = f"custom-{_slug(body_name)}"
    bodies = catalogue.setdefault("bodies", [])
    body = next((item for item in bodies if str(item.get("id") or "").casefold() == body_id.casefold()), None)
    if body is None:
        body = {"id": body_id, "display_name": body_name, "collection": collection, "slots": {}}
        bodies.append(body)
    else:
        body["display_name"] = body_name
        body["collection"] = collection
        body.setdefault("slots", {})

    new_payload_bytes = {}
    imported_slots = set()
    imported_payload_ids = set()
    by_slot = {}
    for row, raw, ref, surface, payload_id, analysis_id in selected:
        slot = row["slot"]
        race_code = row["race_code"]
        imported_slots.add(slot)
        imported_payload_ids.add(payload_id)
        new_payload_bytes[payload_id] = raw
        variant_id = f"{body_id}.{slot.casefold()}.{_slug(variant_name)}"
        selected_by = {"entry_id": variant_id, "body": body_name, "slot": slot, "variant": variant_name, "race_code": race_code}
        source_label = " / ".join(value for value in (source_group, source_option) if value) or variant_name
        source = {"package": source_mod, "label": source_label, "source": row["relative_path"], "target": row["game_path"], "race_code": race_code, "slot": slot}
        payload_meta = {
            "id": payload_id,
            "raw_mdl_sha256": payload_id,
            "solver_identity_sha256": hash_solver_identity(ref),
            "mdl_version": int(ref["mdl_version"]),
            "vertex_count": int(len(ref["positions"])),
            "triangle_count": int(len(ref["indices"]) // 3),
            "body_surface_vertex_count": int(len(surface["V"])),
            "body_surface_triangle_count": int(len(surface["F"])),
            "joint_count": int(len(ref["joint_names"])),
            "joint_names": list(ref["joint_names"]),
            "joint_name_sha256": hashlib.sha256("\0".join(ref["joint_names"]).encode("utf-8")).hexdigest(),
            "solver_compatible": bool(ref["solver_compatible"]),
            "has_normals": bool(any(record.get("has_normal") for record in ref["mesh_records"])),
            "has_uv0": bool(any(record.get("has_uv0") for record in ref["mesh_records"])),
            "has_skin": bool(any(record.get("has_skin") for record in ref["mesh_records"])),
            "bounds_min": ref["bounds_min"],
            "bounds_max": ref["bounds_max"],
            "materials": list(ref["materials"]),
            "body_surface_materials": [record["material"] for record in surface.get("mesh_records", [])],
            "mesh_records": list(ref["mesh_records"]),
            "warning_count": int(ref.get("warning_count", 0)),
            "warnings": list(ref.get("warnings") or []),
            "canonical_source": source,
            "selected_by": [selected_by],
        }
        existing_meta = payload_index.get(payload_id)
        if existing_meta is not None:
            if str(existing_meta.get("raw_mdl_sha256") or payload_id) != payload_id:
                raise ValueError(f"Custom RBODY payload metadata collision for {payload_id}")
            seen = {(str(item.get("entry_id")), str(item.get("race_code"))) for item in (existing_meta.get("selected_by") or [])}
            if (variant_id, race_code) not in seen:
                existing_meta.setdefault("selected_by", []).append(selected_by)
        else:
            payload_index[payload_id] = payload_meta
        by_slot.setdefault(slot, []).append({
            "race_code": race_code,
            "solver_payload_id": payload_id,
            "replacement_payload_id": payload_id,
            "analysis_geometry_id": analysis_id,
            "alternate_model_sha256s": [],
            "source_label": source_label,
            "source_package": source_mod,
            "target_path": row["game_path"],
        })

    for slot, imported_races in by_slot.items():
        variant_id = f"{body_id}.{slot.casefold()}.{_slug(variant_name)}"
        entries = body["slots"].setdefault(slot, [])
        variant = next((entry for entry in entries if str(entry.get("id") or "").casefold() == variant_id.casefold()), None)
        if variant is None:
            variant = {"id": variant_id, "display_name": variant_name, "slot": slot, "race_payloads": []}
            entries.append(variant)
        variant["display_name"] = variant_name
        variant["slot"] = slot
        existing_races = {str(row.get("race_code")): row for row in (variant.get("race_payloads") or [])}
        for row in imported_races:
            existing_races[row["race_code"]] = row
        race_payloads = [existing_races[key] for key in sorted(existing_races)]
        variant["race_payloads"] = race_payloads
        variant["source_packages"] = sorted(set([source_mod] + list(variant.get("source_packages") or [])), key=str.casefold)
        variant["source_labels"] = sorted(set([row["source_label"] for row in race_payloads]), key=str.casefold)
        sexes = sorted(set(filter(None, (_sex_for_race_code(row["race_code"]) for row in race_payloads))))
        variant["sexes"] = sexes
        support_label = " ".join((variant_name, source_group, source_option)).casefold()
        if slot == "Legs" and "male" in sexes and (re.search(r"\bsfw\b", support_label) is not None or "underwear" in support_label):variant["support_surface"] = "smallclothes"
        else:variant.pop("support_surface", None)
        canonical = next((row for row in race_payloads if row["race_code"] == "0201"), None) or next((row for row in race_payloads if row["race_code"] == "0101"), None) or race_payloads[0]
        variant["canonical_solver_payload_id"] = canonical["solver_payload_id"]
        variant["canonical_replacement_payload_id"] = canonical["replacement_payload_id"]
        variant["analysis_geometry_id"] = canonical["analysis_geometry_id"]
        entries.sort(key=lambda entry: str(entry.get("display_name") or "").casefold())

    bodies.sort(key=lambda item: str(item.get("display_name") or "").casefold())
    referenced = {
        str(race.get("solver_payload_id"))
        for item in bodies
        for variants in (item.get("slots") or {}).values()
        for variant in variants or []
        for race in (variant.get("race_payloads") or [])
        if race.get("solver_payload_id")
    }
    payload_index = {pid: meta for pid, meta in payload_index.items() if pid in referenced}
    catalogue["body_count"] = len(bodies)
    catalogue["variant_count"] = sum(len(values) for item in bodies for values in (item.get("slots") or {}).values())
    manifest.update({"body_count": catalogue["body_count"], "variant_count": catalogue["variant_count"], "payload_count": len(payload_index)})

    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
            out.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            out.writestr("catalogue.json", json.dumps(catalogue, indent=2, ensure_ascii=False))
            out.writestr("payload_index.json", json.dumps(payload_index, indent=2, ensure_ascii=False))
            out.writestr("target_options.json", json.dumps({"profiles": {}}, indent=2))
            for payload_id in sorted(referenced):
                if payload_id in new_payload_bytes:
                    raw = new_payload_bytes[payload_id]
                elif old_archive is not None:
                    raw = old_archive.read(f"payloads/{payload_id}.mdl")
                else:
                    raise FileNotFoundError(f"Missing existing custom RBODY payload {payload_id}")
                if hashlib.sha256(raw).hexdigest() != payload_id:
                    raise IOError(f"Custom RBODY payload hash mismatch for {payload_id}")
                out.writestr(f"payloads/{payload_id}.mdl", raw)
        if old_archive is not None:
            old_archive.close()
            old_archive = None
        os.replace(temporary, output)
    finally:
        if old_archive is not None:
            old_archive.close()
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "ok": True,
        "path": str(output),
        "body_name": body_name,
        "variant_name": variant_name,
        "imported_slots": len(imported_slots),
        "imported_payloads": len(imported_payload_ids),
        "body_count": catalogue["body_count"],
        "variant_count": catalogue["variant_count"],
    }


def _prepare_cache(payload):
    from prepare_b14_rbody_cache import rig_names
    from rbody_v3_loader import get_cached_rbody
    from rbody_b14_adapter import collect_body_pairs
    import numpy as np

    rig_glb = Path(payload["rig_glb"]).resolve()
    output = Path(payload["output"]).resolve()
    pairs_spec = payload["pairs"]
    names = rig_names(rig_glb, payload.get("rig_mesh"))
    pairs = []
    for p in pairs_spec:
        sp = str(Path(p["source_rbody"]).resolve())
        tp = str(Path(p["target_rbody"]).resolve())
        source_race = p.get("source_race_code", p.get("race_code"))
        target_race = p.get("target_race_code", p.get("race_code"))
        source = get_cached_rbody(sp).reference(p["source_body"], p["slot"], p["source_variant"], race_code=source_race, rig_joint_names=names, surface_mode=str(p.get("source_support_surface") or "body"))
        target = get_cached_rbody(tp).reference(p["target_body"], p["slot"], p["target_variant"], race_code=target_race, rig_joint_names=names, surface_mode=str(p.get("target_support_surface") or "body"))
        pairs.append((source, target))
    cache = collect_body_pairs(pairs)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=cache["X"], Y=cache["Y"], BW=cache["BW"], target_correspondence_W=cache["target_correspondence_W"], NS=cache["NS"], NT=cache["NT"],
        names=np.asarray(cache["names"], dtype=object), parts=cache["parts"],
        source_surface_V=cache["source_surface_V"], source_surface_F=cache["source_surface_F"], source_surface_W=cache["source_surface_W"],
        target_surface_V=cache["target_surface_V"], target_surface_F=cache["target_surface_F"], target_surface_W=cache["target_surface_W"],
    )
    return {"ok": True, "output": str(output), "joint_count": len(names), "slot_stats": cache["slot_stats"]}


def _extract_payload(payload):
    from rbody_v3_loader import get_cached_rbody
    source = Path(payload["rbody"]).resolve()
    destination = Path(payload["destination"]).resolve()
    race = payload.get("race_code")
    library = get_cached_rbody(source)
    entry = library.entry(payload["body"], payload["slot"], payload["variant"])
    payload_id = library.resolve_payload_id(payload["body"], payload["slot"], payload["variant"], race_code=race)
    model_view = entry.get("model_view") or {}
    if model_view:
        from rbody_v3_core import bake_model_view_mdl
        raw = library.raw_mdl(payload_id)
        viewed, view_report = bake_model_view_mdl(raw, model_view)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(viewed)
        out = destination
    else:
        out = library.write_pristine_mdl(destination, payload["body"], payload["slot"], payload["variant"], race_code=race)
        view_report = {"applied": False}
    target_path = None
    if race is not None:
        for row in entry.get("race_payloads", []):
            if row.get("race_code") == str(race) and row.get("solver_payload_id") == payload_id:
                target_path = row.get("target_path")
                break
    if not target_path:
        for row in entry.get("race_payloads", []):
            if row.get("solver_payload_id") == payload_id:
                target_path = row.get("target_path")
                break
    if not target_path:
        target_path = library.payload_meta(payload_id).get("canonical_source", {}).get("target")
    return {"ok": True, "path": str(out), "payload_id": payload_id, "race_code": str(race) if race is not None else None, "target_path": target_path, "model_view": view_report}



def _analyze_coverage(payload):
    from coverage_analysis import analyse_coverage

    glb = Path(payload["glb"]).resolve()
    if not glb.exists():
        raise FileNotFoundError(glb)
    return analyse_coverage(glb, str(payload["game_path"]))




def _detect_source_bodies(payload):
    from coverage_analysis import detect_source_bodies

    glb = Path(payload["glb"]).resolve()
    if not glb.exists():
        raise FileNotFoundError(glb)
    return detect_source_bodies(
        glb,
        str(payload["game_path"]),
        list(payload.get("candidates") or []),
        list(payload.get("required_slots") or []),
        str(payload["preferred_race_code"]) if payload.get("preferred_race_code") is not None else None,
        dict(payload.get("source_context") or {}),
        str(payload["primary_slot"]) if payload.get("primary_slot") is not None else None,
    )



def _graft_native_body(payload):
    from native_body_graft import graft_native_body

    imported_mdl = Path(payload["imported_mdl"]).resolve()
    output_mdl = Path(payload["output_mdl"]).resolve()
    conversion_report = Path(payload["conversion_report"]).resolve()
    target_body_mdls = {str(slot): str(Path(path).resolve()) for slot, path in (payload.get("target_body_mdls") or {}).items()}
    if not target_body_mdls:
        raise ValueError("Native body graft requires target_body_mdls.")
    source_model_mdl = payload.get("source_model_mdl")
    return graft_native_body(
        imported_mdl, output_mdl, conversion_report, target_body_mdls,
        Path(source_model_mdl).resolve() if source_model_mdl else None,
    )



def _inspect_mdl_parts(payload):
    from customise_mod import inspect_mdl_parts
    return inspect_mdl_parts(Path(payload["mdl"]).resolve())


def _hide_mdl_parts(payload):
    from customise_mod import hide_mdl_parts
    return hide_mdl_parts(Path(payload["mdl"]).resolve(), Path(payload["output"]).resolve(), list(payload.get("part_indices") or []))


def _tag_mdl_parts(payload):
    from customise_mod import tag_mdl_parts
    return tag_mdl_parts(Path(payload["mdl"]).resolve(), Path(payload["output"]).resolve(), list(payload.get("part_indices") or []), str(payload.get("attribute") or ""))


def _split_mdl_parts(payload):
    from customise_mod import split_mdl_parts
    return split_mdl_parts(
        Path(payload["mdl"]).resolve(),
        Path(payload["remaining_output"]).resolve(),
        Path(payload["accessory_output"]).resolve(),
        list(payload.get("part_indices") or []),
    )


def _extract_piercings(payload):
    from customise_mod import extract_piercings_glb
    return extract_piercings_glb(Path(payload["source_glb"]).resolve(), Path(payload["output_glb"]).resolve())


def _replace_piercings(payload):
    from customise_mod import replace_piercings_glb
    return replace_piercings_glb(Path(payload["source_glb"]).resolve(), Path(payload["donor_glb"]).resolve(), Path(payload["output_glb"]).resolve())

def _convert(payload):
    production = _load_production_module()
    spec_path = Path(payload["spec"]).resolve()
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return production.convert(spec)


METHODS = {
    "health": lambda payload: _health(),
    "catalogue": _catalogue,
    "import_body": _import_body,
    "prepare_cache": _prepare_cache,
    "extract_payload": _extract_payload,
    "analyze_coverage": _analyze_coverage,
    "detect_source_bodies": _detect_source_bodies,
    "convert": _convert,
    "graft_native_body": _graft_native_body,
    "inspect_mdl_parts": _inspect_mdl_parts,
    "hide_mdl_parts": _hide_mdl_parts,
    "tag_mdl_parts": _tag_mdl_parts,
    "split_mdl_parts": _split_mdl_parts,
    "extract_piercings": _extract_piercings,
    "replace_piercings": _replace_piercings,
}


def main():
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req_id = None
        try:
            req = json.loads(raw)
            req_id = req.get("id")
            method = req.get("method")
            if method not in METHODS:
                raise KeyError(f"Unknown solver method: {method}")
            result = METHODS[method](req.get("payload") or {})
            _response(req_id, **result)
        except Exception as ex:
            traceback.print_exc(file=sys.stderr)
            frames = traceback.extract_tb(ex.__traceback__)
            if frames:
                frame = frames[-1]
                location = f" [{Path(frame.filename).name}:{frame.lineno} in {frame.name}]"
            else:
                location = ""
            _response(req_id, error=f"{type(ex).__name__}: {ex}{location}", ok=False)


def _health_cli() -> int:
    """Build-time health entry point that does not depend on stdin."""
    try:
        result = _health()
        _response(1, **result)
        return 0
    except BaseException as ex:
        traceback.print_exc(file=sys.stderr)
        _response(1, error=f"{type(ex).__name__}: {ex}", ok=False)
        return 1


if __name__ == "__main__":
    if "--health-json" in sys.argv[1:]:
        raise SystemExit(_health_cli())
    main()
