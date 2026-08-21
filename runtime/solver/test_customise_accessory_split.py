#!/usr/bin/env python3
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import customise_mod as customise


def _fake_parser(_raw: bytes):
    submeshes = [
        {"part_index": 0, "local_index_offset": 0, "index_count": 3, "attributes": [], "attribute_mask": 0},
        {"part_index": 1, "local_index_offset": 3, "index_count": 3, "attributes": [], "attribute_mask": 0},
        {"part_index": 2, "local_index_offset": 6, "index_count": 3, "attributes": [], "attribute_mask": 0},
        {"part_index": 3, "local_index_offset": 9, "index_count": 0, "attributes": ["atrx_empty"], "attribute_mask": 0},
    ]
    mesh = {
        "mesh_index": 0,
        "index_offset": 0,
        "source_index_offset": 0,
        "index_count": 9,
        "material_index": 0,
        "submeshes": submeshes,
    }
    positions = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0],
        [4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 1.0, 0.0],
    ], dtype=np.float32)
    return {
        "materials": ["garment.mtrl"],
        "mesh_records": [mesh],
        "lod_standard_mesh_records": [{"lod": 0, "index_buffer_offset": 32, "records": [mesh]}],
        "positions": positions,
        "normals": np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (9, 1)),
        "uv0": np.zeros((9, 2), dtype=np.float32),
        "indices": np.arange(9, dtype=np.uint32),
        "bounds_min": [0.0, 0.0, 0.0],
        "bounds_max": [5.0, 1.0, 0.0],
    }


def _write_source(path: Path):
    raw = bytearray(64)
    struct.pack_into("<9H", raw, 32, *range(9))
    path.write_bytes(raw)
    return bytes(raw)


def _indices(path: Path):
    raw = path.read_bytes()
    return struct.unpack_from("<9H", raw, 32)


def test_split_keeps_multiple_selected_parts_together_and_source_untouched():
    original_parser = customise._rbody_core
    customise._rbody_core = lambda: _fake_parser
    try:
        with tempfile.TemporaryDirectory(prefix="ravafit-accessory-split-") as temp:
            root = Path(temp)
            source = root / "source.mdl"
            remaining = root / "remaining.mdl"
            accessory = root / "accessory.mdl"
            original = _write_source(source)

            report = customise.split_mdl_parts(source, remaining, accessory, [0, 2])

            assert source.read_bytes() == original
            assert report["selected_parts"] == [0, 2]
            assert report["remaining_parts"] == [1]
            assert len(remaining.read_bytes()) == len(original)
            assert len(accessory.read_bytes()) == len(original)
            # Remaining/source clone hides selected parts 0 + 2, leaving part 1 authored.
            assert _indices(remaining) == (0, 0, 0, 3, 4, 5, 6, 6, 6)
            # Accessory hides only the unselected part 1, keeping both selected parts together.
            assert _indices(accessory) == (0, 1, 2, 3, 3, 3, 6, 7, 8)
    finally:
        customise._rbody_core = original_parser


def test_split_refuses_to_empty_source_model():
    original_parser = customise._rbody_core
    customise._rbody_core = lambda: _fake_parser
    try:
        with tempfile.TemporaryDirectory(prefix="ravafit-accessory-split-") as temp:
            root = Path(temp)
            source = root / "source.mdl"
            _write_source(source)
            try:
                customise.split_mdl_parts(source, root / "remaining.mdl", root / "accessory.mdl", [0, 1, 2])
            except ValueError as ex:
                assert "leave the source equipment model empty" in str(ex)
            else:
                raise AssertionError("split should refuse to move every authored part")
    finally:
        customise._rbody_core = original_parser



def test_split_refuses_currently_empty_authored_part():
    original_parser = customise._rbody_core
    customise._rbody_core = lambda: _fake_parser
    try:
        with tempfile.TemporaryDirectory(prefix="ravafit-accessory-split-") as temp:
            root = Path(temp)
            source = root / "source.mdl"
            _write_source(source)
            inspection = customise.inspect_mdl_parts(source)
            assert 3 not in [int(part["part_index"]) for part in inspection["parts"]]
            try:
                customise.split_mdl_parts(source, root / "remaining.mdl", root / "accessory.mdl", [3])
            except ValueError as ex:
                assert "no longer valid" in str(ex)
            else:
                raise AssertionError("split should refuse an authored part with no visible LOD0 triangles")
    finally:
        customise._rbody_core = original_parser

def main():
    test_split_keeps_multiple_selected_parts_together_and_source_untouched()
    test_split_refuses_to_empty_source_model()
    test_split_refuses_currently_empty_authored_part()
    print("customise accessory split tests: PASS")


if __name__ == "__main__":
    main()
