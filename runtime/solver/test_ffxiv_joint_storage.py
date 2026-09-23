#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import native_body_graft as graft
import production_b14 as prod


class _Editor:
    def __init__(self, js: dict, binary: bytes = b""):
        self.js = js
        self.bin = bytearray(binary)

    def accessor(self, index: int):
        accessor = self.js["accessors"][index]
        view = self.js["bufferViews"][accessor["bufferView"]]
        dtype = {5121: np.uint8, 5123: np.uint16, 5126: np.float32}[accessor["componentType"]]
        components = {"VEC4": 4}[accessor["type"]]
        offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        return np.frombuffer(self.bin, dtype=np.dtype(dtype).newbyteorder("<"), count=int(accessor["count"]) * components, offset=offset).reshape(int(accessor["count"]), components)


def test_gltf_joint_widening_updates_payload_and_accessor_type():
    source = _Editor(
        {"accessors": [{"bufferView": 0, "componentType": 5121, "count": 1, "type": "VEC4"}], "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 4}]},
        bytes([0, 1, 2, 3]),
    )
    target = _Editor({"accessors": [], "bufferViews": []})
    accessor = prod._clone_accessor(target, source, 0, {}, {0: 0, 1: 1, 2: 300, 3: 301})
    assert target.js["accessors"][accessor]["componentType"] == 5123
    assert np.frombuffer(target.bin, dtype="<u2").tolist() == [0, 1, 300, 301]


def test_xiv_mesh_bone_budget_accepts_64_and_rejects_65():
    editor = _Editor({"accessors": [], "bufferViews": [], "meshes": [{"name": "mesh 0", "primitives": []}]})
    joints = np.arange(64, dtype=np.uint8).reshape(16, 4)
    weights = np.ones((16, 4), dtype=np.float32)
    for values, component_type in ((joints, 5121), (weights, 5126)):
        while len(editor.bin) % 4:
            editor.bin.append(0)
        offset = len(editor.bin)
        editor.bin.extend(values.tobytes())
        editor.js["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": values.nbytes})
        editor.js["accessors"].append({"bufferView": len(editor.js["bufferViews"]) - 1, "componentType": component_type, "count": 16, "type": "VEC4"})
    editor.js["meshes"][0]["primitives"] = [{"attributes": {"JOINTS_0": 0, "WEIGHTS_0": 1}}]
    assert prod._validate_xiv_weighted_bone_budget(editor)["meshes"][0]["weighted_joints"] == 64

    editor.bin.extend(b"\0" * 8)
    extra_joint_offset = len(editor.bin) - 8
    extra_joints = np.asarray([[64, 0, 0, 0]], dtype=np.uint8)
    extra_weights = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    editor.bin[extra_joint_offset:extra_joint_offset + 4] = extra_joints.tobytes()
    editor.bin[extra_joint_offset + 4:extra_joint_offset + 8] = b"\0" * 4
    # A separate primitive is enough to prove the union is enforced across the XIV mesh.
    editor.js["bufferViews"].append({"buffer": 0, "byteOffset": extra_joint_offset, "byteLength": 4})
    editor.js["accessors"].append({"bufferView": len(editor.js["bufferViews"]) - 1, "componentType": 5121, "count": 1, "type": "VEC4"})
    while len(editor.bin) % 4:
        editor.bin.append(0)
    weight_offset = len(editor.bin)
    editor.bin.extend(extra_weights.tobytes())
    editor.js["bufferViews"].append({"buffer": 0, "byteOffset": weight_offset, "byteLength": extra_weights.nbytes})
    editor.js["accessors"].append({"bufferView": len(editor.js["bufferViews"]) - 1, "componentType": 5126, "count": 1, "type": "VEC4"})
    editor.js["meshes"][0]["primitives"].append({"attributes": {"JOINTS_0": 2, "WEIGHTS_0": 3}})
    try:
        prod._validate_xiv_weighted_bone_budget(editor)
    except ValueError as ex:
        assert "at most 64" in str(ex)
    else:
        raise AssertionError("65 weighted joints should have been rejected")


def test_unused_native_bone_slots_are_not_treated_as_weighted():
    declaration = bytearray(136)
    declaration[0:5] = bytes([0, 0, 8, 1, 0])
    declaration[8:13] = bytes([0, 4, 5, 2, 0])
    declaration[16] = 255
    model = {
        "infos": [[{"stride": [8, 0, 0], "vdo": [0, 0, 0], "vcount": 1, "bone_set": 0}]],
        "declarations": [bytes(declaration)],
        "data": bytearray([255, 0, 0, 0, 0, 1, 2, 3]),
        "lods": [{"voff": 0}],
        "bone_sets": [[0, 1, 2, 3]],
        "joint_names": ["j_kosi", "j_kao", "j_oya_a_l", "j_x"],
    }
    assert graft._weighted_local_bone_ordinals_for_mesh(model, 0) == {0}
    assert graft._weighted_bone_names_for_mesh(model, 0) == {"j_kosi"}


def test_native_body_mapping_accepts_penumbra_index_rebuild_by_material_lineage():
    original = graft._penumbra_normalised_index_topology
    graft._penumbra_normalised_index_topology = lambda model, mesh: {
        "block": b"", "part_offsets": [0], "part_counts": [0], "dropped": []
    }
    try:
        def info(vcount, icount, mat):
            return {"vcount": vcount, "icount": icount, "mat": mat, "part_count": 1, "idxoff": 0,
                    "parts": [{"index_offset": 0, "index_count": icount}]}

        output = {
            "infos": [[info(99, 12, 0), info(10, 6, 1), info(20, 9, 1), info(30, 12, 1)]],
            "materials": ["/garment.mtrl", "/body.mtrl"],
        }
        native = {
            "infos": [[info(10, 9, 0), info(1, 3, 0), info(1, 3, 0), info(1, 3, 0), info(1, 3, 0), info(20, 12, 0), info(30, 15, 0)]],
            "materials": ["/body.mtrl"],
        }
        native_models = {"native.mdl": native}
        mapping = [
            {"slot": "Legs", "reported_output_mesh": 0, "output_mesh": 0, "native_mesh": 0, "native_mdl": "native.mdl"},
            {"slot": "Legs", "reported_output_mesh": 5, "output_mesh": 5, "native_mesh": 5, "native_mdl": "native.mdl"},
            {"slot": "Legs", "reported_output_mesh": 6, "output_mesh": 6, "native_mesh": 6, "native_mdl": "native.mdl"},
        ]
        resolved, details = graft._resolve_reconstructed_body_mapping(output, mapping, native_models)
        assert [row["output_mesh"] for row in resolved] == [1, 2, 3]
        assert [row["reported_output_mesh"] for row in resolved] == [0, 5, 6]
        assert all(item["compacted"] for item in details)
        assert graft._body_topology_candidate(output, 0, native, 0) is False
    finally:
        graft._penumbra_normalised_index_topology = original


def main():
    test_gltf_joint_widening_updates_payload_and_accessor_type()
    test_xiv_mesh_bone_budget_accepts_64_and_rejects_65()
    test_unused_native_bone_slots_are_not_treated_as_weighted()
    test_native_body_mapping_accepts_penumbra_index_rebuild_by_material_lineage()
    print("FFXIV joint storage tests: PASS")


if __name__ == "__main__":
    main()

