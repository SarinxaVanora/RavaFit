from __future__ import annotations

import json
import struct
from pathlib import Path

import native_body_graft as graft


def _declaration() -> bytes:
    block = bytearray(136)
    block[0:5] = bytes([0, 0, 2, 0, 0])  # POSITION / FLOAT3 / stream 0
    block[8] = 255
    return bytes(block)


def _write_mdl(path: Path, meshes: list[dict], materials: list[str]) -> None:
    mesh_count = len(meshes)
    material_count = len(materials)
    strings = ["j_kosi", *materials]
    path_block = b"".join(value.encode() + b"\0" for value in strings)
    model_values = [
        mesh_count, 0, mesh_count, material_count, 1, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0,
    ]
    model_data = struct.pack(graft.MODEL_DATA_FMT, *model_values)
    metadata_size = (
        68 + 136 * mesh_count + 8 + len(path_block) + 4 + len(model_data)
        + 3 * 60 + mesh_count * 36 + mesh_count * 16
        + material_count * 4 + 4 + 8
    )

    vertex_buffer = bytearray()
    index_buffer = bytearray()
    infos = []
    parts = []
    for part_index, mesh in enumerate(meshes):
        vertex_offset = len(vertex_buffer)
        for position in mesh["positions"]:
            vertex_buffer.extend(struct.pack("<3f", *position))
        index_offset = len(index_buffer) // 2
        for value in mesh["indices"]:
            index_buffer.extend(struct.pack("<H", value))
        infos.append((
            mesh["vcount"], mesh["icount"], mesh["mat"], part_index, 1, 0,
            index_offset, vertex_offset, 0, 0, 12, 0, 0, 1,
        ))
        parts.append((index_offset, mesh["icount"], 0, 0, 1))

    vertex_offset = metadata_size
    index_offset = vertex_offset + len(vertex_buffer)
    lod0 = struct.pack(
        "<HHffHHHHHHHH8i", 0, mesh_count, 0.0, 0.0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, len(vertex_buffer), len(index_buffer), vertex_offset, index_offset,
    )
    empty_lod = struct.pack(
        "<HHffHHHHHHHH8i", 0, 0, 0.0, 0.0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    )

    data = bytearray(68)
    data.extend(b"".join(_declaration() for _ in meshes))
    data.extend(struct.pack("<ii", 0, len(path_block)))
    data.extend(path_block)
    data.extend(struct.pack("<f", 1.0))
    data.extend(model_data)
    data.extend(lod0)
    data.extend(empty_lod)
    data.extend(empty_lod)
    data.extend(b"".join(struct.pack("<iihhhhiiiiBBBB", *row) for row in infos))
    data.extend(b"".join(struct.pack("<IIIHH", *row) for row in parts))
    data.extend(b"\0" * (material_count * 4))
    data.extend(b"\0" * 4)
    data.extend(struct.pack("<hh", 0, 1))
    data.extend(struct.pack("<h", 0))
    data.extend(b"\0\0")
    assert len(data) == metadata_size
    data.extend(vertex_buffer)
    data.extend(index_buffer)

    struct.pack_into("<I", data, 0, 6)
    struct.pack_into("<H", data, 12, mesh_count)
    struct.pack_into("<H", data, 14, material_count)
    struct.pack_into("<I", data, 16, vertex_offset)
    struct.pack_into("<I", data, 28, index_offset)
    struct.pack_into("<I", data, 40, len(vertex_buffer))
    struct.pack_into("<I", data, 52, len(index_buffer))
    data[64] = 1
    path.write_bytes(data)


def test_native_body_graft_rebuilds_compacted_index_topology_without_touching_garment(tmp_path: Path):
    garment = {"vcount": 3, "icount": 3, "mat": 0, "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "indices": [0, 1, 2]}
    reconstructed_body = {"vcount": 3, "icount": 3, "mat": 1, "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "indices": [0, 1, 2]}
    native_body = {"vcount": 3, "icount": 6, "mat": 0, "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "indices": [0, 1, 2, 0, 2, 1]}

    imported = tmp_path / "imported.mdl"
    native = tmp_path / "native.mdl"
    output = tmp_path / "output.mdl"
    report = tmp_path / "report.json"
    _write_mdl(imported, [garment, reconstructed_body], ["/garment.mtrl", "/body.mtrl"])
    _write_mdl(native, [native_body], ["/body.mtrl"])
    report.write_text(json.dumps({"transplant": {"inserted": [{"slot": "Legs", "name": "mesh 5", "target_xiv_mesh_index": 0}]}}))

    reply = graft.graft_native_body(imported, output, report, {"Legs": str(native)})
    parsed_imported = graft._parse(imported)
    parsed_native = graft._parse(native)
    parsed_output = graft._parse(output)

    assert reply["output_mesh_remap"] == [{"slot": "Legs", "native_mesh": 0, "reported_output_mesh": 5, "resolved_output_mesh": 1, "compacted": True}]
    assert reply["body_meshes"][0]["topology_mode"] == "native-index-rebuild"
    assert graft._index_block(parsed_output, 0) == graft._index_block(parsed_imported, 0)
    assert graft._index_block(parsed_output, 1) == graft._index_block(parsed_native, 0)
    assert parsed_output["infos"][0][1]["icount"] == 6
