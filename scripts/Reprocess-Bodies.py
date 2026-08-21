#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
import sys
from pathlib import Path
from typing import Any

PIERCING_RE = re.compile(r"pierc|jewel|dermal|barbell|bellyring|nipple", re.I)
SUPPORT_RE = re.compile(r"required|reguired|base\s*install|main\s*files|reqyp", re.I)
GROUP_RE = re.compile(r"(^|/)group_.*\.json$", re.I)

YAP_BODY_IDS = {
    "yab", "yab-mini", "yabulky", "yet-another-masc", "lavabod-plus-2-2", "rue",
}

# Installed labels and archive names do not always match, so keep the aliases explicit.
PACKAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "annabelle-plus": ("Annabelle+.pmp", "Annabelle+ Navel Piercings Add on.pmp"),
    "citrus-plus": ("Citrus+.pmp",),
    "coconut": ("COCONUT - for Rox+ & Citrus+ [YAS]",),
    "dionys": ("Dionys",),
    "kt-plus": ("KT+ Body - Main Installer.pmp", "KT+ Piercing Selector.pmp",),
    "lavabod-plus-2-2": ("LavaBod+ 2.2.pmp", "Yet Another Piercing+.pmp"),
    "lumme": ("lumme",),
    "muse-plus": ("[YAS] Muse+",),
    "neolithe": ("Neolithe [ALL IN ONE]",),
    "rosaline": ("Rosaline (Base)",),
    "rox-plus": ("Rox+",),
    "rue": ("Rue+", "Yet Another Piercing+.pmp"),
    "yab": ("Yet Another Body+.pmp", "Yet Another Piercing+.pmp"),
    "yab-mini": ("YAB 4.0 MINI", "Yet Another Piercing+.pmp"),
    "yabulky": ("YABulky", "Yet Another Piercing+.pmp"),
    "yet-another-masc": ("Yet Another Masc", "Yet Another Piercing+.pmp"),
    "andor": ("Andor _ TBSE body",),
    "tbse-chonk": ("The Body SE-Chonk",),
    "tre": ("Tre (Body)",),
    "moon": ("Moon~ Body",),
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def canonical_rel(s: str) -> str:
    return str(s or "").replace("\\", "/").lstrip("/")


def _rbody_core():
    runtime_rbody = Path(__file__).resolve().parents[1] / "runtime" / "rbody"
    value = str(runtime_rbody)
    if value not in sys.path:
        sys.path.insert(0, value)
    from rbody_v3_core import body_surface_view, hash_solver_identity, parse_rigged_mdl
    return parse_rigged_mdl, body_surface_view, hash_solver_identity


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return text or "variant"


def _analysis_geometry_id(surface: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(surface["V"].astype("<f4", copy=False).tobytes())
    h.update(surface["F"].astype("<i4", copy=False).tobytes())
    return h.hexdigest()


def _payload_meta(raw: bytes, source: dict[str, Any], selected_by: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    parse_rigged_mdl, body_surface_view, hash_solver_identity = _rbody_core()
    pid = hashlib.sha256(raw).hexdigest()
    ref = parse_rigged_mdl(raw)
    surface = body_surface_view(ref)
    analysis_id = _analysis_geometry_id(surface)
    meta = {
        "id": pid,
        "raw_mdl_sha256": pid,
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
        "mesh_records": [dict(record, material=(ref["materials"][index] if index < len(ref["materials"]) else "")) for index, record in enumerate(ref["mesh_records"])],
        "warning_count": int(ref.get("warning_count", 0)),
        "warnings": list(ref.get("warnings") or []),
        "canonical_source": source,
        "selected_by": [selected_by],
    }
    return pid, meta, analysis_id


def _sectioned_options(group: dict[str, Any]):
    section = ""
    for option in [item for item in (group.get("Options") or []) if isinstance(item, dict)]:
        name = str(option.get("Name") or "").strip()
        match = re.fullmatch(r"-+\s*(.*?)\s*-+", name)
        if match:
            section = match.group(1).strip()
            continue
        yield section, option


def _find_group(z: zipfile.ZipFile, wanted_name: str) -> tuple[str, dict[str, Any]]:
    for name in z.namelist():
        if not GROUP_RE.search(canonical_rel(name)):
            continue
        try:
            group = json.loads(z.read(name))
        except Exception:
            continue
        if str(group.get("Name") or "").casefold() == wanted_name.casefold():
            return name, group
    raise RuntimeError(f"Could not find Penumbra group {wanted_name!r} in source package.")


def _option_model(option: dict[str, Any], slot: str) -> tuple[str, str] | None:
    suffix = {"Chest": "_top.mdl", "Legs": "_dwn.mdl"}[slot]
    files = option.get("Files") or {}
    if not isinstance(files, dict):
        return None
    for game_path, rel in files.items():
        if str(game_path).replace("\\", "/").casefold().endswith(suffix):
            return str(game_path).replace("\\", "/"), canonical_rel(str(rel))
    return None


def repair_neolithe_sectioned_variants(catalogue: dict[str, Any], payload_index: dict[str, Any], payload_sources: dict[str, tuple[Path, str]], sources: dict[str, Path]) -> dict[str, Any]:
    """Restore Neolithe variants lost when repeated option names were collapsed."""
    body = next((item for item in catalogue.get("bodies", []) if str(item.get("id")) == "neolithe"), None)
    if body is None:
        return {"repaired": False, "reason": "Neolithe body not present"}
    package = resolve_source(sources, "Neolithe [ALL IN ONE]")
    if package is None:
        return {"repaired": False, "reason": "Neolithe source package not available"}

    rebuilt: dict[str, list[dict[str, Any]]] = {"Chest": [], "Legs": []}
    with zipfile.ZipFile(package) as z:
        lookup = zip_lookup(z)
        _, chest_group = _find_group(z, "CHEST: SmallClothes")
        _, legs_group = _find_group(z, "LEGS: SmallClothes")

        def add_variant(slot: str, display_name: str, option: dict[str, Any], source_group_name: str):
            model = _option_model(option, slot)
            if model is None:
                return
            game_path, rel = model
            actual = lookup.get(rel.casefold())
            if actual is None:
                raise RuntimeError(f"Neolithe source model is missing from package: {rel}")
            raw = z.read(actual)
            race_match = re.search(r"(?:^|/)c([0-9]{4})[^/]*\.mdl$", game_path, re.I)
            race_code = race_match.group(1) if race_match else "0201"
            entry_id = f"neolithe.{slot.casefold()}.{_slug(display_name)}"
            label = f"{source_group_name} / {option.get('Name')} [{display_name}]"
            selected_by = {"entry_id": entry_id, "body": "Neolithe", "slot": slot, "variant": display_name, "race_code": race_code}
            source = {"package": "Neolithe [ALL IN ONE]", "label": label, "source": actual, "target": game_path, "race_code": race_code, "slot": slot}
            pid, meta, analysis_id = _payload_meta(raw, source, selected_by)
            if pid not in payload_index:
                payload_index[pid] = meta
                payload_sources[pid] = (package, actual)
            else:
                existing = payload_index[pid]
                seen = {(row.get("entry_id"), row.get("race_code")) for row in existing.get("selected_by") or []}
                key = (entry_id, race_code)
                if key not in seen:
                    existing.setdefault("selected_by", []).append(selected_by)
            rebuilt[slot].append({
                "id": entry_id,
                "display_name": display_name,
                "slot": slot,
                "source_packages": ["Neolithe [ALL IN ONE]"],
                "source_labels": [label],
                "sexes": ["female"],
                "race_payloads": [{
                    "race_code": race_code,
                    "solver_payload_id": pid,
                    "replacement_payload_id": pid,
                    "analysis_geometry_id": analysis_id,
                    "alternate_model_sha256s": [],
                    "source_label": label,
                    "source_package": "Neolithe [ALL IN ONE]",
                    "target_path": game_path,
                }],
                "canonical_solver_payload_id": pid,
                "canonical_replacement_payload_id": pid,
                "analysis_geometry_id": analysis_id,
            })

        for section, option in _sectioned_options(chest_group):
            raw_name = str(option.get("Name") or "").strip()
            if not raw_name.casefold().startswith("nsfw "):
                continue
            base = raw_name[5:].strip()
            sec = section.upper()
            if sec.startswith("DEFAULT"):
                display = base
            elif sec.startswith("NEOBELLY"):
                display = base if base.casefold().startswith("neobelly ") else f"Neobelly {base}"
            elif sec.startswith("BUFF"):
                display = base if base.casefold().startswith("buff ") else f"Buff {base}"
            elif sec == "FLAT":
                display = base
            else:
                continue
            add_variant("Chest", display, option, str(chest_group.get("Name") or "CHEST: SmallClothes"))

        for section, option in _sectioned_options(legs_group):
            raw_name = str(option.get("Name") or "").strip()
            sec = section.upper()
            if sec not in {"DEFAULT", "NEOBELLY"} or not raw_name.casefold().startswith("gen "):
                continue
            display = raw_name if sec == "DEFAULT" else f"Neobelly {raw_name}"
            add_variant("Legs", display, option, str(legs_group.get("Name") or "LEGS: SmallClothes"))

    body.setdefault("slots", {})["Chest"] = sorted(rebuilt["Chest"], key=lambda x: x["display_name"].casefold())
    old_legs = list(body.get("slots", {}).get("Legs") or [])
    retained_legs = [entry for entry in old_legs if not re.match(r"^(?:Neobelly\s+)?Gen\s+(?:A|B|C|Puffy)\s+", str(entry.get("display_name") or ""), re.I)]
    body["slots"]["Legs"] = sorted(retained_legs + rebuilt["Legs"], key=lambda x: x["display_name"].casefold())
    catalogue["variant_count"] = sum(len(values) for item in catalogue.get("bodies", []) for values in (item.get("slots") or {}).values())
    return {"repaired": True, "chest_variants": len(rebuilt["Chest"]), "legs_sectioned_variants": len(rebuilt["Legs"]), "total_neolithe_legs": len(body["slots"]["Legs"])}


def source_index(roots: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for p in paths:
            if p.is_file() and p.suffix.casefold() in {".pmp", ".ttmp2", ".zip"}:
                out[norm(p.name)] = p
    return out


def resolve_source(index: dict[str, Path], alias: str) -> Path | None:
    n = norm(alias)
    if n in index:
        return index[n]
    candidates = [(k, p) for k, p in index.items() if n in k or k in n]
    if not candidates:
        return None
    candidates.sort(key=lambda x: abs(len(x[0]) - len(n)))
    return candidates[0][1]


def is_user_piercing_group(g: dict[str, Any]) -> bool:
    name = str(g.get("Name") or "")
    options = [o for o in (g.get("Options") or []) if isinstance(o, dict)]
    return bool(PIERCING_RE.search(name) or any(PIERCING_RE.search(str(o.get("Name") or "")) for o in options))


def is_piercing_support_group(g: dict[str, Any]) -> bool:
    name = str(g.get("Name") or "")
    return bool(SUPPORT_RE.search(name) and PIERCING_RE.search(json.dumps(g, ensure_ascii=False)))


def group_slots(g: dict[str, Any]) -> list[str]:
    if str(g.get("Type") or "").casefold() == "imc":
        equip = str((g.get("Identifier") or {}).get("EquipSlot") or "").casefold()
        if equip == "body": return ["Chest"]
        if equip in {"legs", "leg"}: return ["Legs"]
        if equip in {"hands", "hand", "gloves"}: return ["Hands"]
        if equip in {"feet", "foot", "shoes"}: return ["Feet"]
    name = str(g.get("Name") or "").casefold()
    if any(t in name for t in ("genital", "bottom", "lower", "hips", "hip ")):
        return ["Legs"]
    if any(t in name for t in ("nipple", "torso", "navel", "upper", "center", "chest", "collar", "belly", "barbell")):
        return ["Chest"]
    # Colour/material support can feed both chest and legs piercing materials.
    return ["Chest", "Legs"]


def zip_lookup(z: zipfile.ZipFile) -> dict[str, str]:
    return {canonical_rel(n).casefold(): n for n in z.namelist() if not n.endswith("/")}


def capture_package(path: Path, assets: dict[str, dict[str, Any]], asset_bytes: dict[str, bytes]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    report = {"package": path.name, "groups": 0, "user_groups": 0, "support_groups": 0, "missing_assets": [], "skipped_model_groups": []}
    with zipfile.ZipFile(path) as z:
        lookup = zip_lookup(z)
        for name in z.namelist():
            if not GROUP_RE.search(canonical_rel(name)):
                continue
            try:
                group = json.loads(z.read(name))
            except Exception:
                continue
            dedicated_piercing_pack = bool(PIERCING_RE.search(path.stem))
            support = is_piercing_support_group(group) or (dedicated_piercing_pack and bool(SUPPORT_RE.search(str(group.get("Name") or ""))))
            meaningful = any(
                isinstance(o, dict) and ((o.get("Files") or {}) or (o.get("FileSwaps") or {}) or (o.get("Manipulations") or []))
                for o in (group.get("Options") or [])
            )
            boilerplate = bool(re.search(r"^(?:final\s+page|finish|credits?)$", str(group.get("Name") or "").strip(), re.I))
            user = (is_user_piercing_group(group) and not support) or (dedicated_piercing_pack and not support and meaningful and not boilerplate)
            if not user and not support:
                continue
            # Reject piercing groups that replace an MDL wholesale.
            model_replacements: list[str] = []
            file_asset_map: dict[str, str] = {}
            for option in [o for o in (group.get("Options") or []) if isinstance(o, dict)]:
                files = option.get("Files") or {}
                if not isinstance(files, dict):
                    continue
                for game_path, rel in files.items():
                    if str(game_path).casefold().endswith(".mdl") or str(rel).casefold().endswith(".mdl"):
                        model_replacements.append(str(game_path))
                        continue
                    rel_norm = canonical_rel(str(rel))
                    actual = lookup.get(rel_norm.casefold())
                    if actual is None:
                        report["missing_assets"].append({"group": group.get("Name"), "relative_path": rel_norm})
                        continue
                    blob = z.read(actual)
                    sha = hashlib.sha256(blob).hexdigest()
                    suffix = Path(rel_norm).suffix.casefold()
                    asset_id = sha + suffix
                    asset_bytes.setdefault(asset_id, blob)
                    assets.setdefault(asset_id, {
                        "sha256": sha,
                        "extension": suffix,
                        "size": len(blob),
                        "source_package": path.name,
                        "source_path": rel_norm,
                    })
                    file_asset_map[rel_norm.casefold()] = asset_id
            if model_replacements:
                report["skipped_model_groups"].append({"group": group.get("Name"), "model_paths": sorted(set(model_replacements))})
                continue
            # Take race scope from game-path keys, not reused source texture paths.
            authoritative_paths: list[str] = []
            for option in [o for o in (group.get("Options") or []) if isinstance(o, dict)]:
                files = option.get("Files") or {}
                swaps = option.get("FileSwaps") or {}
                if isinstance(files, dict): authoritative_paths.extend(str(x) for x in files.keys())
                if isinstance(swaps, dict): authoritative_paths.extend(str(x) for x in swaps.keys())
            race_codes = sorted(set(re.findall(r"(?:^|[/\\])c([0-9]{4})(?:[/\\]|[^0-9])", "\n".join(authoritative_paths), re.I)))
            groups.append({
                "source_package": path.name,
                "source_group_file": canonical_rel(name),
                "kind": "control" if user else "support",
                "slots": group_slots(group),
                "race_codes": race_codes,
                "group": group,
                "file_assets": file_asset_map,
            })
            report["groups"] += 1
            report["user_groups" if user else "support_groups"] += 1
    return groups, report


def merge_catalogues(inputs: list[Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[Path, str]], list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []
    payload_index: dict[str, Any] = {}
    payload_sources: dict[str, tuple[Path, str]] = {}
    omissions: list[dict[str, Any]] = []
    ids: set[str] = set()
    collections: list[str] = []
    for path in inputs:
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json"))
            catalogue = json.loads(z.read("catalogue.json"))
            idx = json.loads(z.read("payload_index.json"))
            collection = str(catalogue.get("collection") or manifest.get("category") or path.stem)
            if collection not in collections: collections.append(collection)
            omissions.extend(manifest.get("known_omissions") or [])
            for body in catalogue.get("bodies") or []:
                b = copy.deepcopy(body)
                b["collection"] = b.get("collection") or collection
                body_id = str(b.get("id") or "unknown")
                if body_id in ids:
                    # Male and female both contain RE[DE]FINED Lalafell; keep them separate.
                    body_id = f"{norm(collection)}-{body_id}"
                    b["id"] = body_id
                    for slot_values in (b.get("slots") or {}).values():
                        for variant in slot_values or []:
                            old = str(variant.get("id") or "")
                            if old:
                                variant["id"] = f"{norm(collection)}-{old}"
                ids.add(body_id)
                bodies.append(b)
            for pid, meta in idx.items():
                if pid in payload_index:
                    if payload_index[pid].get("raw_mdl_sha256") != meta.get("raw_mdl_sha256"):
                        raise RuntimeError(f"Payload collision for {pid}")
                    existing = payload_index[pid]
                    seen = {(x.get("entry_id"), x.get("race_code")) for x in existing.get("selected_by") or []}
                    for row in meta.get("selected_by") or []:
                        key = (row.get("entry_id"), row.get("race_code"))
                        if key not in seen:
                            existing.setdefault("selected_by", []).append(row); seen.add(key)
                    continue
                payload_index[pid] = meta
                entry = f"payloads/{pid}.mdl"
                if entry not in z.namelist():
                    raise RuntimeError(f"{path.name} is missing {entry}")
                payload_sources[pid] = (path, entry)
    catalogue = {
        "format": "RBODY",
        "version": 4,
        "collection": "RavaFit Bodies",
        "collections": collections,
        "slot_order": ["Chest", "Legs", "Hands", "Feet"],
        "body_count": len(bodies),
        "variant_count": sum(len(v) for b in bodies for v in (b.get("slots") or {}).values()),
        "bodies": bodies,
        "payload_semantics": "Immutable pristine source MDLs; body-level collection preserves the legacy library grouping.",
    }
    return catalogue, payload_index, payload_sources, omissions




def repair_rosaline_sectioned_variants(catalogue: dict[str, Any], payload_index: dict[str, Any], payload_sources: dict[str, tuple[Path, str]], sources: dict[str, Path]) -> dict[str, Any]:
    """Restore Rosaline lower-body variants with their section identity."""
    body = next((item for item in catalogue.get("bodies", []) if str(item.get("id")) == "rosaline"), None)
    package = resolve_source(sources, "Rosaline (Base)")
    if body is None or package is None:
        return {"repaired": False, "reason": "Rosaline body/source unavailable"}
    rebuilt: list[dict[str, Any]] = []
    with zipfile.ZipFile(package) as z:
        lookup = zip_lookup(z)
        _, group = _find_group(z, "Smallclothes Legs")
        for section, option in _sectioned_options(group):
            if section.strip().casefold() not in {"hip dips", "round hips"}:
                continue
            files = option.get("Files") or {}
            if not isinstance(files, dict):
                continue
            for game_path, rel in files.items():
                game_path = canonical_rel(str(game_path))
                if "/e0000/" not in f"/{game_path.casefold()}" or not game_path.casefold().endswith("_dwn.mdl"):
                    continue
                rel_norm = canonical_rel(str(rel))
                actual = lookup.get(rel_norm.casefold())
                if actual is None:
                    raise RuntimeError(f"Rosaline source model missing: {rel_norm}")
                raw = z.read(actual)
                race_match = re.search(r"(?:^|/)c([0-9]{4})[^/]*\.mdl$", game_path, re.I)
                race_code = race_match.group(1) if race_match else "0201"
                section_name = _normalise_variant_label(section).title()
                option_name = _normalise_variant_label(str(option.get("Name") or ""))
                display = f"{section_name} {option_name}".strip()
                entry_id = f"rosaline.legs.{_slug(display)}"
                label = f"{group.get('Name') or 'Smallclothes Legs'} / {section} / {option.get('Name')}"
                selected_by = {"entry_id": entry_id, "body": "Rosaline", "slot": "Legs", "variant": display, "race_code": race_code}
                source = {"package": package.stem, "label": label, "source": actual, "target": game_path, "race_code": race_code, "slot": "Legs"}
                pid, meta, analysis_id = _payload_meta(raw, source, selected_by)
                if pid not in payload_index:
                    payload_index[pid] = meta
                    payload_sources[pid] = (package, actual)
                else:
                    existing = payload_index[pid]
                    seen = {(x.get("entry_id"), x.get("race_code")) for x in existing.get("selected_by") or []}
                    if (entry_id, race_code) not in seen:
                        existing.setdefault("selected_by", []).append(selected_by)
                rebuilt.append({
                    "id": entry_id,
                    "display_name": display,
                    "slot": "Legs",
                    "source_packages": [package.stem],
                    "source_labels": [label],
                    "sexes": ["female"],
                    "race_payloads": [{
                        "race_code": race_code,
                        "solver_payload_id": pid,
                        "replacement_payload_id": pid,
                        "analysis_geometry_id": analysis_id,
                        "alternate_model_sha256s": [],
                        "source_label": label,
                        "source_package": package.stem,
                        "target_path": game_path,
                    }],
                    "canonical_solver_payload_id": pid,
                    "canonical_replacement_payload_id": pid,
                    "analysis_geometry_id": analysis_id,
                })
    if len(rebuilt) != 16:
        raise RuntimeError(f"Rosaline section repair expected 16 authored lower-body choices, found {len(rebuilt)}")
    body.setdefault("slots", {})["Legs"] = sorted(rebuilt, key=lambda x: x["display_name"].casefold())
    catalogue["variant_count"] = sum(len(values) for item in catalogue.get("bodies", []) for values in (item.get("slots") or {}).values())
    return {"repaired": True, "legs_variants": len(rebuilt), "sections": ["Hip Dips", "Round Hips"], "source_package": package.name}

def _slot_from_model_path(game_path: str) -> str | None:
    value = canonical_rel(game_path).casefold()
    for suffix, slot in (("_top.mdl", "Chest"), ("_dwn.mdl", "Legs"), ("_glv.mdl", "Hands"), ("_sho.mdl", "Feet")):
        if value.endswith(suffix):
            return slot
    return None


def _source_package_paths_for_body(body: dict[str, Any], sources: dict[str, Path]) -> list[Path]:
    """Resolve the original body installers represented by a catalogue body."""
    aliases = [str(x) for x in (body.get("source_packages") or []) if str(x).strip()]
    # Installed names can include Heliosphere decoration that is not in the archive name.
    aliases.extend(PACKAGE_ALIASES.get(str(body.get("id")), ()))
    resolved: list[Path] = []
    seen: set[Path] = set()
    for alias in aliases:
        path = resolve_source(sources, alias)
        if path is not None and path not in seen:
            resolved.append(path)
            seen.add(path)
    return resolved


def _normalise_variant_label(value: str) -> str:
    text = re.sub(r"\s*-\s*", " ", str(value or "").strip())
    return re.sub(r"\s+", " ", text).strip()


def _package_variant_prefix(package: Path, body: dict[str, Any]) -> str:
    """Return the useful framework flavour carried by the package identity."""
    name = package.stem.casefold()
    body_id = str(body.get("id") or "")
    parts: list[str] = []
    if "ivcs" in name:
        parts.append("IVCS")
    if "yas" in name or "[yas]" in name:
        parts.append("YAS")
    if "legacy" in name:
        parts.append("Legacy")
    # Muse comes from the YAS source package even when the option label does not say so.
    if body_id == "muse-plus" and "YAS" not in parts and "yas" in name:
        parts.append("YAS")
    return " ".join(parts)


def _source_option_rows(z: zipfile.ZipFile, package: Path):
    """Yield selectable e0000 model choices with section context and source bytes."""
    lookup = zip_lookup(z)
    for group_file in z.namelist():
        if not GROUP_RE.search(canonical_rel(group_file)):
            continue
        try:
            group = json.loads(z.read(group_file))
        except Exception:
            continue
        option_rows = list(_sectioned_options(group))
        counts: dict[str, int] = {}
        for _, option in option_rows:
            key = str(option.get("Name") or "").strip().casefold()
            counts[key] = counts.get(key, 0) + 1
        for section, option in option_rows:
            option_name = str(option.get("Name") or "").strip()
            files = option.get("Files") or {}
            if not isinstance(files, dict):
                continue
            for game_path, rel in files.items():
                game_path = canonical_rel(str(game_path))
                if "/e0000/" not in f"/{game_path.casefold()}":
                    continue
                slot = _slot_from_model_path(game_path)
                if slot is None:
                    continue
                rel_norm = canonical_rel(str(rel))
                actual = lookup.get(rel_norm.casefold())
                if actual is None:
                    continue
                raw = z.read(actual)
                race_match = re.search(r"(?:^|/)c([0-9]{4})[^/]*\.mdl$", game_path, re.I)
                race_code = race_match.group(1) if race_match else "0201"
                yield {
                    "package": package,
                    "group_file": canonical_rel(group_file),
                    "group_name": str(group.get("Name") or ""),
                    "section": section,
                    "section_required": counts.get(option_name.casefold(), 0) > 1 and bool(section.strip()),
                    "option_name": option_name,
                    "slot": slot,
                    "game_path": game_path,
                    "source_path": actual,
                    "race_code": race_code,
                    "raw": raw,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }


def _promoted_display_name(body: dict[str, Any], package: Path, row: dict[str, Any], existing_names: set[str]) -> str:
    base = _normalise_variant_label(row["option_name"])
    if row.get("section_required"):
        base = f"{_normalise_variant_label(row.get('section') or '')} {base}".strip()
    prefix = _package_variant_prefix(package, body)
    if prefix:
        prefix_tokens = [token.casefold() for token in prefix.split() if token]
        base_words = {token.casefold() for token in re.findall(r"[A-Za-z0-9+]+", base)}
        if not all(token in base_words for token in prefix_tokens):
            base = f"{prefix} {base}".strip()
    # Only add a package-flavour suffix when the authored label would otherwise collide.
    if base.casefold() in existing_names and not prefix:
        stem = package.stem
        if str(body.get("id")) == "annabelle-plus":
            stem = "Base"
        discriminator = _normalise_variant_label(stem)
        body_name = _normalise_variant_label(body.get("display_name") or "")
        if body_name and discriminator.casefold().startswith(body_name.casefold()):
            discriminator = discriminator[len(body_name):].strip(" +-_[]()") or "Base"
        base = f"{discriminator} {base}".strip()
    return base


def promote_source_alternate_variants(catalogue: dict[str, Any], payload_index: dict[str, Any], payload_sources: dict[str, tuple[Path, str]], sources: dict[str, Path]) -> dict[str, Any]:
    """Promote source-authored choices previously stored only as alternate hashes."""
    promoted: list[dict[str, Any]] = []
    package_reports: list[dict[str, Any]] = []
    for body in catalogue.get("bodies", []):
        slots = body.get("slots") or {}
        alt_owners: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}
        for slot, variants in slots.items():
            for variant in variants or []:
                for race_payload in variant.get("race_payloads") or []:
                    for alt in list(race_payload.get("alternate_model_sha256s") or []):
                        alt_owners.setdefault(str(alt), []).append((slot, variant, race_payload))
        if not alt_owners:
            continue
        packages = _source_package_paths_for_body(body, sources)
        if not packages:
            continue

        existing_names_by_slot = {
            slot: {str(v.get("display_name") or "").casefold() for v in variants or []}
            for slot, variants in slots.items()
        }
        candidates: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        recovered_hashes: set[str] = set()
        for package in packages:
            try:
                z = zipfile.ZipFile(package)
            except Exception:
                continue
            with z:
                for row in _source_option_rows(z, package):
                    owners = alt_owners.get(row["sha256"])
                    if not owners:
                        continue
                    # Promote an alternate only into the exact slot/race/body lane that recorded it.
                    matches = [owner for owner in owners if owner[0] == row["slot"] and str(owner[2].get("race_code") or "0201") == row["race_code"]]
                    if not matches:
                        continue
                    display = _promoted_display_name(body, package, row, existing_names_by_slot.setdefault(row["slot"], set()))
                    key = (row["slot"], display.casefold(), str(package), row["group_file"], row["option_name"] + "\0" + str(row.get("section") or ""))
                    candidates.setdefault(key, []).append(row)
                    recovered_hashes.add(row["sha256"])

        body_promoted = 0
        for key, rows in sorted(candidates.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
            slot = key[0]
            # De-dupe identical race/hash rows from the same authored option.
            unique_rows: list[dict[str, Any]] = []
            seen_rows: set[tuple[str, str]] = set()
            for row in rows:
                row_key = (row["race_code"], row["sha256"])
                if row_key not in seen_rows:
                    unique_rows.append(row); seen_rows.add(row_key)
            if not unique_rows:
                continue
            package = unique_rows[0]["package"]
            display = _promoted_display_name(body, package, unique_rows[0], existing_names_by_slot.setdefault(slot, set()))
            base_display = display
            suffix = 2
            while display.casefold() in existing_names_by_slot.setdefault(slot, set()):
                group_hint = _normalise_variant_label(unique_rows[0].get("group_name") or "")
                candidate = f"{base_display} ({group_hint})" if suffix == 2 and group_hint else f"{base_display} {suffix}"
                display = candidate
                suffix += 1
            existing_names_by_slot[slot].add(display.casefold())
            entry_id = f"{body['id']}.{slot.casefold()}.{_slug(display)}"
            existing_ids = {str(v.get("id") or "") for v in slots.get(slot, [])}
            id_suffix = 2
            base_id = entry_id
            while entry_id in existing_ids:
                entry_id = f"{base_id}-{id_suffix}"; id_suffix += 1

            race_payloads: list[dict[str, Any]] = []
            source_labels: list[str] = []
            source_packages: list[str] = []
            sexes: set[str] = set()
            for row in unique_rows:
                owners = [owner for owner in alt_owners.get(row["sha256"], []) if owner[0] == slot and str(owner[2].get("race_code") or "0201") == row["race_code"]]
                if not owners:
                    continue
                owner_variant = owners[0][1]
                sexes.update(str(x) for x in (owner_variant.get("sexes") or []))
                label_parts = [str(row.get("group_name") or "")]
                if row.get("section"):
                    label_parts.append(str(row["section"]))
                label_parts.append(str(row.get("option_name") or ""))
                label = " / ".join(x for x in label_parts if x)
                package_name = row["package"].stem
                source_packages.append(package_name)
                source_labels.append(label)
                selected_by = {"entry_id": entry_id, "body": body.get("display_name") or body["id"], "slot": slot, "variant": display, "race_code": row["race_code"]}
                source = {"package": package_name, "label": label, "source": row["source_path"], "target": row["game_path"], "race_code": row["race_code"], "slot": slot}
                pid, meta, analysis_id = _payload_meta(row["raw"], source, selected_by)
                if pid not in payload_index:
                    payload_index[pid] = meta
                    payload_sources[pid] = (row["package"], row["source_path"])
                else:
                    existing = payload_index[pid]
                    seen_selection = {(x.get("entry_id"), x.get("race_code")) for x in existing.get("selected_by") or []}
                    if (entry_id, row["race_code"]) not in seen_selection:
                        existing.setdefault("selected_by", []).append(selected_by)
                race_payloads.append({
                    "race_code": row["race_code"],
                    "solver_payload_id": pid,
                    "replacement_payload_id": pid,
                    "analysis_geometry_id": analysis_id,
                    "alternate_model_sha256s": [],
                    "source_label": label,
                    "source_package": package_name,
                    "target_path": row["game_path"],
                })
                for _, _, owner_race in owners:
                    owner_race["alternate_model_sha256s"] = [h for h in (owner_race.get("alternate_model_sha256s") or []) if str(h) != row["sha256"]]

            if not race_payloads:
                continue
            race_payloads.sort(key=lambda x: str(x.get("race_code") or ""))
            first = race_payloads[0]
            variant = {
                "id": entry_id,
                "display_name": display,
                "slot": slot,
                "source_packages": sorted(set(source_packages), key=str.casefold),
                "source_labels": sorted(set(source_labels), key=str.casefold),
                "sexes": sorted(sexes),
                "race_payloads": race_payloads,
                "canonical_solver_payload_id": first["solver_payload_id"],
                "canonical_replacement_payload_id": first["replacement_payload_id"],
                "analysis_geometry_id": first["analysis_geometry_id"],
            }
            slots.setdefault(slot, []).append(variant)
            promoted.append({
                "body": body.get("display_name") or body.get("id"),
                "body_id": body.get("id"),
                "slot": slot,
                "display_name": display,
                "source_package": package.name,
                "source_group": unique_rows[0].get("group_name"),
                "source_section": unique_rows[0].get("section"),
                "source_option": unique_rows[0].get("option_name"),
                "race_codes": [r["race_code"] for r in race_payloads],
                "payload_sha256s": [r["replacement_payload_id"] for r in race_payloads],
            })
            body_promoted += 1
        for slot in slots:
            slots[slot] = sorted(slots[slot], key=lambda x: str(x.get("display_name") or "").casefold())
        if body_promoted:
            package_reports.append({"body": body.get("display_name") or body.get("id"), "promoted_variants": body_promoted, "recovered_alternate_hashes": len(recovered_hashes), "source_packages": [p.name for p in packages]})

    catalogue["variant_count"] = sum(len(values) for item in catalogue.get("bodies", []) for values in (item.get("slots") or {}).values())
    remaining_alt_refs = sum(len(rp.get("alternate_model_sha256s") or []) for body in catalogue.get("bodies", []) for values in (body.get("slots") or {}).values() for variant in values or [] for rp in (variant.get("race_payloads") or []))
    remaining_alt_unique = len({h for body in catalogue.get("bodies", []) for values in (body.get("slots") or {}).values() for variant in values or [] for rp in (variant.get("race_payloads") or []) for h in (rp.get("alternate_model_sha256s") or [])})
    return {
        "promoted_variant_count": len(promoted),
        "promoted": promoted,
        "by_body": package_reports,
        "remaining_alternate_refs": remaining_alt_refs,
        "remaining_alternate_unique": remaining_alt_unique,
    }


def refresh_tbse_chonk_ivcs_from_source(catalogue: dict[str, Any], payload_index: dict[str, Any], payload_sources: dict[str, tuple[Path, str]], sources: dict[str, Path]) -> dict[str, Any]:
    """Refresh TBSE-Chonk IVCS torso payloads from the current installer."""
    body = next((item for item in catalogue.get("bodies", []) if str(item.get("id")) == "tbse-chonk"), None)
    package = resolve_source(sources, "The Body SE-Chonk")
    if body is None or package is None:
        return {"refreshed": False, "reason": "TBSE-Chonk body/source unavailable"}
    rows: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(package) as z:
        for row in _source_option_rows(z, package):
            if row["slot"] != "Chest" or str(row.get("option_name") or "").casefold() != "tbse-chonk ivcs":
                continue
            rows[row["race_code"]] = row
    if not rows:
        return {"refreshed": False, "reason": "Current TBSE-Chonk IVCS source rows not found"}

    refreshed: list[dict[str, Any]] = []
    for variant in body.get("slots", {}).get("Chest", []) or []:
        for rp in variant.get("race_payloads") or []:
            source_label = str(rp.get("source_label") or "")
            race = str(rp.get("race_code") or "")
            row = rows.get(race)
            if row is None or "tbse-chonk ivcs" not in source_label.casefold():
                continue
            old_pid = str(rp.get("replacement_payload_id") or "")
            label = f"{row.get('group_name') or ''} / {row.get('option_name') or ''}".strip(" / ")
            selected_by = {"entry_id": variant.get("id"), "body": body.get("display_name") or body["id"], "slot": "Chest", "variant": variant.get("display_name"), "race_code": race}
            source = {"package": package.stem, "label": label, "source": row["source_path"], "target": row["game_path"], "race_code": race, "slot": "Chest"}
            pid, meta, analysis_id = _payload_meta(row["raw"], source, selected_by)
            if pid not in payload_index:
                payload_index[pid] = meta
                payload_sources[pid] = (package, row["source_path"])
            rp.update({
                "solver_payload_id": pid,
                "replacement_payload_id": pid,
                "analysis_geometry_id": analysis_id,
                "alternate_model_sha256s": [],
                "source_label": label,
                "source_package": package.stem,
                "target_path": row["game_path"],
            })
            variant["canonical_solver_payload_id"] = pid
            variant["canonical_replacement_payload_id"] = pid
            variant["analysis_geometry_id"] = analysis_id
            variant["source_packages"] = [package.stem]
            variant["source_labels"] = [label]
            refreshed.append({"race_code": race, "variant": variant.get("display_name"), "old_payload": old_pid, "new_payload": pid})
    return {"refreshed": bool(refreshed), "rows": refreshed, "source_package": package.name}


def prune_unreferenced_payloads(catalogue: dict[str, Any], payload_index: dict[str, Any], payload_sources: dict[str, tuple[Path, str]]) -> dict[str, int]:
    referenced = {
        str(rp[key])
        for body in catalogue.get("bodies", [])
        for values in (body.get("slots") or {}).values()
        for variant in values or []
        for rp in (variant.get("race_payloads") or [])
        for key in ("solver_payload_id", "replacement_payload_id")
        if rp.get(key)
    }
    before = len(payload_index)
    for pid in list(payload_index):
        if pid not in referenced:
            payload_index.pop(pid, None)
            payload_sources.pop(pid, None)
    return {"before": before, "after": len(payload_index), "removed": before - len(payload_index)}


def repair_variant_display_labels(catalogue: dict[str, Any]) -> dict[str, Any]:
    """Repair labels where legacy normalisation removed framework identity."""
    changes: list[dict[str, str]] = []
    for body in catalogue.get("bodies", []):
        body_id = str(body.get("id") or "")
        for slot, variants in (body.get("slots") or {}).items():
            used = {str(v.get("display_name") or "").casefold() for v in variants or []}
            for variant in variants or []:
                old = str(variant.get("display_name") or "")
                new = old
                races = variant.get("race_payloads") or []
                source_package = str((races[0] if races else {}).get("source_package") or "")
                source_label = str((races[0] if races else {}).get("source_label") or "")
                option_label = _normalise_variant_label(source_label.rsplit(" / ", 1)[-1]) if source_label else old
                if body_id == "muse-plus" and "[yas] muse+" in source_package.casefold() and not old.casefold().startswith("yas "):
                    new = f"YAS {option_label}"
                elif body_id == "annabelle-plus":
                    low = source_package.casefold()
                    prefix = ""
                    if "ivcs" in low and "legacy" in low: prefix = "IVCS Legacy"
                    elif "ivcs" in low: prefix = "IVCS"
                    elif "yas" in low: prefix = "YAS"
                    elif "legacy" in low: prefix = "Legacy"
                    if prefix and not old.casefold().startswith(prefix.casefold() + " "):
                        new = f"{prefix} {option_label}"
                elif body_id in {"citrus-plus", "rox-plus"} and old == "&" and "yas" in source_label.casefold():
                    new = "YAS"
                if new != old:
                    base = new
                    suffix = 2
                    used.discard(old.casefold())
                    while new.casefold() in used:
                        new = f"{base} {suffix}"; suffix += 1
                    variant["display_name"] = new
                    used.add(new.casefold())
                    changes.append({"body": str(body.get("display_name") or body_id), "slot": slot, "from": old, "to": new})
        for slot in body.get("slots") or {}:
            body["slots"][slot] = sorted(body["slots"][slot], key=lambda x: str(x.get("display_name") or "").casefold())
    return {"change_count": len(changes), "changes": changes}

def profile_packages(body: dict[str, Any], idx: dict[str, Path]) -> list[Path]:
    explicit = PACKAGE_ALIASES.get(str(body.get("id")))
    # Explicit body mappings do not fall through to fuzzy installed-label matching.
    aliases = list(explicit) if explicit is not None else [str(x) for x in (body.get("source_packages") or [])]
    out: list[Path] = []
    seen: set[Path] = set()
    for alias in aliases:
        p = resolve_source(idx, alias)
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out


def build_options(catalogue: dict[str, Any], sources: dict[str, Path]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    profiles: dict[str, Any] = {}
    assets: dict[str, dict[str, Any]] = {}
    asset_bytes: dict[str, bytes] = {}
    reports: list[dict[str, Any]] = []
    package_cache: dict[Path, list[dict[str, Any]]] = {}
    for body in catalogue["bodies"]:
        packages = profile_packages(body, sources)
        all_groups: list[dict[str, Any]] = []
        user_count = 0
        for package in packages:
            if package not in package_cache:
                groups, report = capture_package(package, assets, asset_bytes)
                package_cache[package] = groups
                reports.append(report)
            groups = package_cache[package]
            all_groups.extend(copy.deepcopy(groups))
            user_count += sum(1 for g in groups if g["kind"] == "control")
        if user_count == 0:
            continue
        unique: list[dict[str, Any]] = []
        seen = set()
        for g in all_groups:
            key = (g["source_package"].casefold(), g["source_group_file"].casefold())
            if key not in seen:
                unique.append(g); seen.add(key)
        sem_unique: list[dict[str, Any]] = []
        imc_seen: dict[str, int] = {}
        for g in unique:
            node = g["group"]
            if str(node.get("Type") or "").casefold() != "imc":
                sem_unique.append(g); continue
            sig_node = copy.deepcopy(node)
            sig_node.pop("Name", None); sig_node.pop("Description", None); sig_node.pop("Page", None); sig_node.pop("Priority", None)
            ident = sig_node.get("Identifier") or {}
            if isinstance(ident, dict): ident["PrimaryId"] = 0
            signature = json.dumps(sig_node, sort_keys=True, ensure_ascii=False)
            if signature in imc_seen:
                existing = sem_unique[imc_seen[signature]]
                old_id = int(((existing["group"].get("Identifier") or {}).get("PrimaryId") or 0))
                new_id = int(((node.get("Identifier") or {}).get("PrimaryId") or 0))
                if new_id < old_id: sem_unique[imc_seen[signature]] = g
                continue
            imc_seen[signature] = len(sem_unique); sem_unique.append(g)
        unique = sem_unique

        profile_id = str(body["id"])
        profiles[profile_id] = {
            "id": profile_id,
            "body_id": body["id"],
            "body_name": body.get("display_name") or body["id"],
            "collection": body.get("collection"),
            "source_packages": [p.name for p in packages],
            "groups": unique,
        }
        body["target_option_profile_id"] = profile_id
    doc = {
        "format": "RAVAFIT_TARGET_BODY_OPTIONS",
        "version": 1,
        "profile_count": len(profiles),
        "profiles": profiles,
        "asset_count": len(assets),
        "assets": assets,
    }
    return doc, assets, asset_bytes, reports



def normalise_catalogue_sexes(catalogue: dict[str, Any]) -> dict[str, Any]:
    corrected=[]
    for body in catalogue.get("bodies", []):
        for slot,variants in (body.get("slots") or {}).items():
            for variant in variants or []:
                sexes=set()
                for row in variant.get("race_payloads") or []:
                    try:prefix=int(str(row.get("race_code") or "")[:2])
                    except Exception:continue
                    if 1 <= prefix <= 18:sexes.add("male" if prefix % 2 else "female")
                if sexes and sorted(str(x).casefold() for x in (variant.get("sexes") or [])) != sorted(sexes):
                    variant["sexes"]=sorted(sexes);corrected.append(str(variant.get("id") or ""))
        known={str(v).casefold() for variants in (body.get("slots") or {}).values() for variant in variants or [] for v in (variant.get("sexes") or [])}
        collection=str(body.get("collection") or "").casefold()
        if known=={"female"} and collection=="male":body["collection"]="Female"
        elif known=={"male"} and collection=="female":body["collection"]="Male"
    return {"corrected":len(corrected),"variants":sorted(corrected)}

def annotate_gender_port_support(catalogue: dict[str, Any]) -> dict[str, Any]:
    """Tag authored male SFW legs at build time; runtime never guesses from labels."""
    tagged=[]
    for body in catalogue.get("bodies", []):
        for slot,variants in (body.get("slots") or {}).items():
            for variant in variants or []:
                variant.pop("support_surface",None)
                sexes={str(value).casefold() for value in (variant.get("sexes") or [])}
                if slot != "Legs" or "male" not in sexes:continue
                labels=" ".join([str(variant.get("display_name") or ""),*[str(value) for value in (variant.get("source_labels") or [])]]).casefold()
                if variant.get("sfw") is True or re.search(r"\bsfw\b", labels) is not None or "underwear" in labels:
                    variant["support_surface"]="smallclothes";tagged.append(str(variant.get("id") or ""))
    return {"count":len(tagged),"variants":sorted(tagged)}

def validate_catalogue_integrity(catalogue: dict[str, Any], payload_index: dict[str, Any], payload_sources: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    """Validate the unified catalogue before writing it."""
    body_ids: set[str] = set()
    variant_ids: set[str] = set()
    referenced_payloads: set[str] = set()
    duplicate_display_names: list[dict[str, str]] = []

    for body in catalogue.get("bodies", []):
        body_id = str(body.get("id") or "")
        folded_body_id = body_id.casefold()
        if not body_id or folded_body_id in body_ids:
            raise ValueError(f"Duplicate or empty RBODY body id: {body_id!r}")
        body_ids.add(folded_body_id)

        for slot, variants in (body.get("slots") or {}).items():
            seen_names: set[str] = set()
            for variant in variants or []:
                variant_id = str(variant.get("id") or "")
                folded_variant_id = variant_id.casefold()
                if not variant_id or folded_variant_id in variant_ids:
                    raise ValueError(f"Duplicate or empty RBODY variant id: {variant_id!r}")
                variant_ids.add(folded_variant_id)

                display_name = str(variant.get("display_name") or "")
                folded_name = display_name.casefold()
                if not display_name or folded_name in seen_names:
                    duplicate_display_names.append({"body": str(body.get("display_name") or body_id), "slot": str(slot), "display_name": display_name})
                seen_names.add(folded_name)

                race_payloads = list(variant.get("race_payloads") or [])
                race_ids = {str(row.get("solver_payload_id") or "") for row in race_payloads}
                canonical = str(variant.get("canonical_solver_payload_id") or "")
                if canonical and canonical not in race_ids:
                    raise ValueError(f"{body_id}/{slot}/{variant_id} canonical payload {canonical} is not one of its race payloads")
                for row in race_payloads:
                    for key in ("solver_payload_id", "replacement_payload_id"):
                        payload_id = str(row.get(key) or "")
                        if not payload_id:
                            continue
                        referenced_payloads.add(payload_id)
                        if payload_id not in payload_index:
                            raise ValueError(f"{body_id}/{slot}/{variant_id} references missing payload {payload_id}")
                        if payload_id not in payload_sources:
                            raise ValueError(f"{body_id}/{slot}/{variant_id} payload {payload_id} has no source bytes")

    if duplicate_display_names:
        raise ValueError(f"Duplicate visible target names within a body/slot: {duplicate_display_names}")
    unreferenced = sorted(set(payload_index) - referenced_payloads)
    if unreferenced:
        raise ValueError(f"Unified RBODY still contains {len(unreferenced)} unreferenced payload(s)")
    return {
        "body_ids_unique": True,
        "variant_ids_unique": True,
        "display_names_unique_within_body_slot": True,
        "referenced_payloads": len(referenced_payloads),
        "unreferenced_payloads": 0,
    }

def write_library(output: Path, input_paths: list[Path], source_roots: list[Path]) -> dict[str, Any]:
    catalogue, payload_index, payload_sources, omissions = merge_catalogues(input_paths)
    sources = source_index(source_roots)
    neolithe_repair = repair_neolithe_sectioned_variants(catalogue, payload_index, payload_sources, sources)
    rosaline_repair = repair_rosaline_sectioned_variants(catalogue, payload_index, payload_sources, sources)
    alternate_repair = promote_source_alternate_variants(catalogue, payload_index, payload_sources, sources)
    tbse_chonk_refresh = refresh_tbse_chonk_ivcs_from_source(catalogue, payload_index, payload_sources, sources)
    label_repair = repair_variant_display_labels(catalogue)
    payload_prune = prune_unreferenced_payloads(catalogue, payload_index, payload_sources)
    sex_normalisation = normalise_catalogue_sexes(catalogue)
    gender_port_support = annotate_gender_port_support(catalogue)
    options, assets, asset_bytes, reports = build_options(catalogue, sources)
    validation = validate_catalogue_integrity(catalogue, payload_index, payload_sources)
    manifest = {
        "format": "RBODY",
        "version": 4,
        "category": "RavaFit Bodies",
        "collections": catalogue["collections"],
        "slots": catalogue["slot_order"],
        "catalogue_file": "catalogue.json",
        "payload_index_file": "payload_index.json",
        "target_options_file": "target_options.json",
        "payload_directory": "payloads/",
        "target_option_asset_directory": "target_option_assets/",
        "payload_count": len(payload_index),
        "body_count": catalogue["body_count"],
        "variant_count": catalogue["variant_count"],
        "target_option_profile_count": options["profile_count"],
        "target_option_asset_count": len(assets),
        "known_omissions": omissions,
        "immutability": "Payload MDLs are pristine source bytes. Consumers MUST clone/extract before modifying.",
        "target_options": "Captured from original Penumbra mod-package groups. Runtime copies option assets into the converted outfit and conditions generated body controls on the generated outfit option.",
        "gender_port_support": "Male cross-gender Legs targets use catalogue variants tagged support_surface=smallclothes.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if temp.exists(): temp.unlink()
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as out:
        out.writestr("README.txt", "RavaFit unified RBODY v4. Contains Bibo/Female/Male/Gen3 catalogues, immutable MDLs, and captured target-body option metadata/assets.\n")
        out.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        out.writestr("catalogue.json", json.dumps(catalogue, indent=2, ensure_ascii=False))
        out.writestr("payload_index.json", json.dumps(payload_index, indent=2, ensure_ascii=False))
        out.writestr("target_options.json", json.dumps(options, indent=2, ensure_ascii=False))
        for pid in sorted(payload_sources):
            src, entry = payload_sources[pid]
            with zipfile.ZipFile(src) as zin:
                out.writestr(f"payloads/{pid}.mdl", zin.read(entry))
        for asset_id in sorted(asset_bytes):
            out.writestr(f"target_option_assets/{asset_id}", asset_bytes[asset_id])
    os.replace(temp, output)
    return {"output": str(output), "manifest": manifest, "capture_reports": reports, "source_archive_count": len(sources), "neolithe_section_repair": neolithe_repair, "rosaline_section_repair": rosaline_repair, "alternate_variant_repair": alternate_repair, "tbse_chonk_source_refresh": tbse_chonk_refresh, "variant_label_repair": label_repair, "payload_prune": payload_prune, "sex_normalisation": sex_normalisation, "gender_port_support": gender_port_support, "validation": validation}


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge legacy RavaFit RBODY libraries and reprocess target-body piercing controls from original Penumbra packages.")
    ap.add_argument("--input", action="append", required=True, type=Path, help="Legacy RBODY input (repeat).")
    ap.add_argument("--sources", action="append", default=[], type=Path, help="Directory/file containing original .pmp/.ttmp2/.zip body sources (repeat).")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    result = write_library(args.output, args.input, args.sources)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
