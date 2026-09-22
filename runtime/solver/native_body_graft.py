from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

MODEL_DATA_FMT = '<8hHBBHBBffHhBBBB6h'
USAGE = {0:'POSITION',1:'BONE_WEIGHT',2:'BONE_INDEX',3:'NORMAL',4:'TEXCOORD',5:'FLOW',6:'BINORMAL',7:'COLOR'}
TYPE = {0:'FLOAT1',1:'FLOAT2',2:'FLOAT3',3:'FLOAT4',5:'UBYTE4',6:'SHORT2',7:'SHORT4',8:'UBYTE4N',9:'SHORT2N',10:'SHORT4N',13:'HALF2',14:'HALF4',17:'UBYTE8'}

MODEL_DATA_KEYS = [
    'MeshCount','AttributeCount','MeshPartCount','MaterialCount','BoneCount','BoneSetCount','ShapeCount','ShapePartCount',
    'ShapeDataCount','LoDCount','Flags1','ElementIdCount','TerrainShadowMeshCount','Flags2','ModelClip','ShadowClip',
    'FurnitureBBoxCount','TerrainShadowPartCount','Flags3','BgChange','BgCrest','NeckMorph','BoneSetSize','Unknown13',
    'Patch72','Unknown15','Unknown16','Unknown17',
]
_MESH_RE = re.compile(r'^mesh\s+(?P<mesh>\d+)(?:[.-](?P<sub>\d+))?$', re.IGNORECASE)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_mesh_name(name: str) -> tuple[int, int] | None:
    match = _MESH_RE.match(str(name or ''))
    return None if match is None else (int(match.group('mesh')), int(match.group('sub') or 0))


def _parse(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = bytearray(path.read_bytes())
    if len(data) < 68:
        raise ValueError(f'MDL is too small: {path}')
    version16 = struct.unpack_from('<H', data, 0)[0]
    mdl_version = 6 if version16 >= 6 else 5
    if mdl_version < 6:
        raise NotImplementedError('Native body graft currently requires FFXIV MDL v6 bone-set layout.')

    mesh_count = struct.unpack_from('<H', data, 12)[0]
    if mesh_count <= 0 or 68 + 136 * mesh_count >= len(data):
        raise ValueError(f'Invalid MDL mesh count {mesh_count}: {path}')

    declarations = []
    for mesh_index in range(mesh_count):
        start = 68 + mesh_index * 136
        declarations.append(bytes(data[start:start + 136]))

    off = 68 + 136 * mesh_count
    _, path_size = struct.unpack_from('<ii', data, off)
    off += 8
    path_block = bytes(data[off:off + path_size])
    off += path_size
    radius = struct.unpack_from('<f', data, off)[0]
    off += 4
    model_data_offset = off
    values = struct.unpack_from(MODEL_DATA_FMT, data, off)
    off += struct.calcsize(MODEL_DATA_FMT)
    model_data = {'Radius': radius, **dict(zip(MODEL_DATA_KEYS, values))}
    off += model_data['ElementIdCount'] * 32

    lods = []
    for _ in range(3):
        lod_offset = off
        std_i, std_c = struct.unpack_from('<HH', data, off); off += 4
        mdl_range, tex_range = struct.unpack_from('<ff', data, off); off += 8
        water = struct.unpack_from('<HH', data, off); off += 4
        shadow = struct.unpack_from('<HH', data, off); off += 4
        terrain = struct.unpack_from('<HH', data, off); off += 4
        fog = struct.unpack_from('<HH', data, off); off += 4
        edge_size, edge_off, u6, u7, vsize, isize, voff, ioff = struct.unpack_from('<8i', data, off); off += 32
        lods.append({
            'std_i': std_i, 'std_c': std_c, 'water': water, 'shadow': shadow, 'terrain': terrain, 'fog': fog,
            'vsize': vsize, 'isize': isize, 'voff': voff, 'ioff': ioff, 'struct_off': lod_offset,
        })

    infos = []
    for lod in lods:
        count = lod['std_c'] + lod['water'][1] + lod['shadow'][1] + lod['fog'][1]
        rows = []
        for _ in range(count):
            struct_offset = off
            vals = struct.unpack_from('<iihhhhiiiiBBBB', data, off); off += 36
            rows.append({
                'vcount': vals[0], 'icount': vals[1], 'mat': vals[2], 'part_index': vals[3], 'part_count': vals[4],
                'bone_set': vals[5], 'idxoff': vals[6], 'vdo': [vals[7], vals[8], vals[9]],
                'stride': [vals[10], vals[11], vals[12]], 'stream_count': vals[13], 'struct_off': struct_offset,
            })
        infos.append(rows)

    off += model_data['AttributeCount'] * 4
    off += lods[0]['terrain'][1] * 20

    # Preserve MeshPart.AttributeMask; piercing/visibility IMC depends on it.
    mesh_parts = []
    for part_index in range(model_data['MeshPartCount']):
        struct_offset = off
        index_offset, index_count, attribute_mask, bone_start, bone_count = struct.unpack_from('<IIIHH', data, off)
        off += 16
        mesh_parts.append({
            'index_offset': index_offset, 'index_count': index_count, 'attribute_mask': attribute_mask,
            'bone_start': bone_start, 'bone_count': bone_count, 'struct_off': struct_offset, 'part_index': part_index,
        })
    for rows in infos:
        for info in rows:
            start = int(info['part_index']); count = int(info['part_count'])
            if start < 0 or count < 0 or start + count > len(mesh_parts):
                raise ValueError(f'Invalid MeshPart range {start}+{count} in {path}.')
            info['parts'] = mesh_parts[start:start + count]
    off += model_data['TerrainShadowPartCount'] * 12
    off += model_data['MaterialCount'] * 4
    off += model_data['BoneCount'] * 4

    bone_set_start = off
    bone_set_end = bone_set_start + model_data['BoneSetSize'] * 2 + model_data['BoneSetCount'] * 4
    bone_sets = []
    bone_set_meta = []
    meta = []
    for _ in range(model_data['BoneSetCount']):
        a, count = struct.unpack_from('<hh', data, off)
        meta.append((a, count, off))
        off += 4
    for _, count, meta_off in meta:
        data_off = off
        values = list(struct.unpack_from('<' + 'h' * count, data, off)) if count else []
        off += 2 * count
        if count % 2:
            off += 2
        bone_sets.append(values)
        bone_set_meta.append({'meta_off': meta_off, 'data_off': data_off, 'count': count})
    off = bone_set_end

    strings = [item.decode('utf-8', 'replace') for item in path_block.split(b'\0') if item]
    attribute_names = strings[:model_data['AttributeCount']]
    if len(attribute_names) != model_data['AttributeCount']:
        raise ValueError(f'Attribute path count mismatch in {path}: {len(attribute_names)} != {model_data["AttributeCount"]}')
    bone_start = model_data['AttributeCount']
    bone_end = bone_start + model_data['BoneCount']
    material_end = bone_end + model_data['MaterialCount']
    joint_names = strings[bone_start:bone_end]
    material_names = strings[bone_end:material_end]

    if len(joint_names) != model_data['BoneCount']:
        raise ValueError(f'Bone path count mismatch in {path}: {len(joint_names)} != {model_data["BoneCount"]}')
    return {
        'path': str(path), 'data': data, 'mdl_version': mdl_version, 'model_data': model_data,
        'model_data_offset': model_data_offset, 'lods': lods, 'infos': infos, 'declarations': declarations,
        'attributes': attribute_names, 'mesh_parts': mesh_parts,
        'joint_names': joint_names, 'materials': material_names, 'bone_sets': bone_sets, 'bone_set_meta': bone_set_meta,
    }


def _bone_names(model: dict[str, Any], mesh_index: int) -> list[str]:
    info = model['infos'][0][mesh_index]
    bone_set_index = int(info['bone_set'])
    if bone_set_index < 0 or bone_set_index >= len(model['bone_sets']):
        raise ValueError(f'Mesh {mesh_index} references invalid bone set {bone_set_index} in {model["path"]}')
    out = []
    for global_index in model['bone_sets'][bone_set_index]:
        if global_index < 0 or global_index >= len(model['joint_names']):
            raise ValueError(f'Mesh {mesh_index} bone set references invalid global joint {global_index} in {model["path"]}')
        out.append(model['joint_names'][global_index])
    return out


def _stream_block(model: dict[str, Any], mesh_index: int, stream: int) -> bytes:
    lod = model['lods'][0]
    info = model['infos'][0][mesh_index]
    stride = int(info['stride'][stream])
    if stride <= 0:
        return b''
    start = lod['voff'] + int(info['vdo'][stream])
    size = int(info['vcount']) * stride
    return bytes(model['data'][start:start + size])


def _index_block(model: dict[str, Any], mesh_index: int) -> bytes:
    lod = model['lods'][0]
    info = model['infos'][0][mesh_index]
    start = lod['ioff'] + int(info['idxoff']) * 2
    return bytes(model['data'][start:start + int(info['icount']) * 2])


def _suppress_index_block(block: bytes, mesh_index: int, triangles: list[int] | tuple[int, ...] | set[int]) -> bytes:
    block = bytearray(block)
    triangle_count = len(block) // 6
    for triangle in sorted({int(value) for value in triangles}):
        if triangle < 0 or triangle >= triangle_count:
            raise ValueError(f'Suppressed native triangle {triangle} is outside mesh {mesh_index} ({triangle_count} triangles).')
        offset = triangle * 6
        first = struct.unpack_from('<H', block, offset)[0]
        struct.pack_into('<HHH', block, offset, first, first, first)
    return bytes(block)


def _suppressed_index_block(model: dict[str, Any], mesh_index: int, triangles: list[int] | tuple[int, ...] | set[int]) -> bytes:
    return _suppress_index_block(_index_block(model, mesh_index), mesh_index, triangles)


def _position_array(model: dict[str, Any], mesh_index: int):
    import numpy as np
    info = model['infos'][0][mesh_index]
    declarations = _parse_decl_elements(model['declarations'][mesh_index])
    positions = next((item for item in declarations if item[3] == 0), None)
    if positions is None:
        raise ValueError(f'Native mesh {mesh_index} has no POSITION declaration in {model["path"]}.')
    block, doff, typ, _, _ = positions
    if block > 2 or int(info['stride'][block]) <= 0:
        raise ValueError(f'Native mesh {mesh_index} has an invalid POSITION stream in {model["path"]}.')
    base = model['lods'][0]['voff'] + int(info['vdo'][block])
    values = np.asarray(_decode_attribute(model['data'], base, int(info['stride'][block]), doff, typ, int(info['vcount'])), dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != int(info['vcount']) or values.shape[1] < 3:
        raise ValueError(f'Native mesh {mesh_index} POSITION data is malformed in {model["path"]}.')
    return values[:, :3]


def _penumbra_normalised_index_topology(model: dict[str, Any], mesh_index: int) -> dict[str, Any]:
    """Mirror the one topology normalisation Penumbra's glTF exporter performs that matters here.

    SharpGLTF/Penumbra omits triangles whose vertices collapse to the exact same POSITION. XIV MDLs
    can legitimately contain those zero-area triangles. They render nothing, but their omission changes
    MeshPart/index counts on an MDL -> glTF -> MDL round trip. Build the canonical native index payload
    after only those exporter-degenerate triangles are removed; any other topology difference remains fatal.
    """
    import numpy as np
    info = model['infos'][0][mesh_index]
    positions = _position_array(model, mesh_index)
    raw = np.frombuffer(_index_block(model, mesh_index), dtype='<u2')
    chunks: list[bytes] = []
    part_counts: list[int] = []
    part_offsets: list[int] = []
    dropped: list[dict[str, int]] = []
    running = 0
    mesh_triangle_base = 0
    for part_ordinal, part in enumerate(info.get('parts') or []):
        relative = int(part['index_offset']) - int(info['idxoff'])
        count = int(part['index_count'])
        if relative < 0 or count < 0 or relative + count > len(raw) or count % 3:
            raise ValueError(
                f'Native mesh {mesh_index} part {part_ordinal} has invalid triangle range {relative}/{count} '
                f'within {len(raw)} indices in {model["path"]}.'
            )
        triangles = raw[relative:relative + count].reshape((-1, 3))
        keep = np.ones(len(triangles), dtype=bool)
        for triangle_ordinal, triangle in enumerate(triangles):
            if np.any(triangle >= len(positions)):
                raise ValueError(f'Native mesh {mesh_index} part {part_ordinal} references a vertex outside {len(positions)} vertices.')
            a, b, c = positions[triangle]
            if np.array_equal(a, b) or np.array_equal(b, c) or np.array_equal(a, c):
                keep[triangle_ordinal] = False
                dropped.append({
                    'part': part_ordinal,
                    'part_triangle': triangle_ordinal,
                    'mesh_triangle': mesh_triangle_base + triangle_ordinal,
                })
        filtered = np.asarray(triangles[keep], dtype='<u2').reshape(-1)
        part_offsets.append(running)
        part_counts.append(int(filtered.size))
        running += int(filtered.size)
        chunks.append(filtered.tobytes())
        mesh_triangle_base += len(triangles)
    return {
        'block': b''.join(chunks),
        'part_offsets': part_offsets,
        'part_counts': part_counts,
        'dropped': dropped,
    }


def _mapping_from_report(report: dict[str, Any], target_body_mdls: dict[str, str]) -> list[dict[str, Any]]:
    inserted = list(((report.get('transplant') or {}).get('inserted') or []))
    if not inserted:
        raise ValueError('Conversion report contains no transplanted body meshes to graft.')

    by_output: dict[int, dict[str, Any]] = {}
    for row in inserted:
        parsed = _parse_mesh_name(row.get('name', ''))
        if parsed is None:
            raise ValueError(f'Conversion report contains an invalid output mesh name: {row.get("name")!r}')

        reported_output_mesh, _ = parsed
        slot = str(row.get('slot') or '')
        if slot not in target_body_mdls:
            raise ValueError(f'No native target MDL was supplied for transplanted slot {slot!r}.')

        native_mesh = int(row['target_xiv_mesh_index'])
        candidate = {
            'slot': slot,
            'reported_output_mesh': reported_output_mesh,
            'output_mesh': reported_output_mesh,
            'native_mesh': native_mesh,
            'native_mdl': str(Path(target_body_mdls[slot]).resolve()),
        }

        existing = by_output.get(reported_output_mesh)
        if existing is not None and existing != candidate:
            raise ValueError(
                f'Reported XIV mesh {reported_output_mesh} maps to conflicting native target meshes: '
                f'{existing} vs {candidate}'
            )

        by_output[reported_output_mesh] = candidate

    return [by_output[key] for key in sorted(by_output)]


def _mesh_part_layout(model: dict[str, Any], mesh_index: int) -> tuple[tuple[int, int], ...]:
    info = model['infos'][0][mesh_index]
    base = int(info['idxoff'])
    return tuple(
        (
            int(part['index_offset']) - base,
            int(part['index_count']),
        )
        for part in (info.get('parts') or [])
    )


def _mesh_material_key(model: dict[str, Any], mesh_index: int) -> str:
    info = model['infos'][0][mesh_index]
    material_index = int(info['mat'])
    if material_index < 0 or material_index >= len(model['materials']):
        return ''

    value = str(model['materials'][material_index] or '').replace('\\', '/')
    return value.rsplit('/', 1)[-1].casefold()


def _body_topology_candidate(
    output: dict[str, Any],
    output_mesh: int,
    native: dict[str, Any],
    native_mesh: int,
) -> bool:
    if output_mesh < 0 or output_mesh >= len(output['infos'][0]):
        return False
    if native_mesh < 0 or native_mesh >= len(native['infos'][0]):
        return False

    oi = output['infos'][0][output_mesh]
    ni = native['infos'][0][native_mesh]

    if int(oi['vcount']) != int(ni['vcount']):
        return False
    if int(oi['part_count']) != int(ni['part_count']):
        return False

    output_layout = _mesh_part_layout(output, output_mesh)
    native_layout = _mesh_part_layout(native, native_mesh)

    if int(oi['icount']) == int(ni['icount']) and output_layout == native_layout:
        return True

    normalised = _penumbra_normalised_index_topology(native, native_mesh)
    normalised_layout = tuple(zip(normalised['part_offsets'], normalised['part_counts']))

    return (
        bool(normalised['dropped'])
        and int(oi['icount']) == len(normalised['block']) // 2
        and output_layout == normalised_layout
    )


def _body_candidate_score(
    output: dict[str, Any],
    output_mesh: int,
    row: dict[str, Any],
    native: dict[str, Any],
) -> tuple[int, int, int, int]:
    native_mesh = int(row['native_mesh'])
    reported = int(row['reported_output_mesh'])

    material_match = int(
        bool(_mesh_material_key(output, output_mesh))
        and _mesh_material_key(output, output_mesh) == _mesh_material_key(native, native_mesh)
    )

    bone_match = 0
    try:
        output_bones = _weighted_bone_names_for_mesh(output, output_mesh)
        native_bones = _weighted_bone_names_for_mesh(native, native_mesh)
        bone_match = int(bool(native_bones) and output_bones == native_bones)
    except Exception:
        # Bone identity is additional disambiguation only. The full graft audit
        # below remains authoritative.
        pass

    exact_reported_index = int(output_mesh == reported)

    # Final tie-breaker only. Penumbra preserves mesh ordering while compacting
    # sparse glTF mesh identities, so the nearest order-preserving position is
    # preferable when otherwise identical native body pieces exist.
    displacement = -abs(output_mesh - reported)

    return material_match, bone_match, exact_reported_index, displacement


def _resolve_reconstructed_body_mapping(
    output: dict[str, Any],
    mapping: list[dict[str, Any]],
    native_models: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not mapping:
        return mapping, []

    ordered = sorted(
        mapping,
        key=lambda row: (
            int(row['reported_output_mesh']),
            str(row['slot']),
            int(row['native_mesh']),
        ),
    )

    candidates: list[list[int]] = []
    for row in ordered:
        native = native_models[row['native_mdl']]
        native_mesh = int(row['native_mesh'])

        matches = [
            output_mesh
            for output_mesh in range(len(output['infos'][0]))
            if _body_topology_candidate(output, output_mesh, native, native_mesh)
        ]

        if not matches:
            raise ValueError(
                f'Native body graft could not locate reconstructed counterpart for reported mesh '
                f'{row["reported_output_mesh"]} ({row["slot"]} native mesh {native_mesh}). '
                f'The imported model contains {len(output["infos"][0])} mesh(es), but none has the '
                'required target-body topology.'
            )

        candidates.append(matches)

    solutions: list[tuple[tuple[int, int, int, int], list[int]]] = []

    def search(position: int, used: set[int], previous: int, chosen: list[int],
               score: tuple[int, int, int, int]) -> None:
        if position == len(ordered):
            solutions.append((score, list(chosen)))
            return

        row = ordered[position]
        native = native_models[row['native_mdl']]

        for output_mesh in candidates[position]:
            if output_mesh in used:
                continue

            # Penumbra may compact sparse mesh identities, but it preserves their
            # relative order. Never solve an ambiguity by swapping body pieces.
            if output_mesh <= previous:
                continue

            part_score = _body_candidate_score(output, output_mesh, row, native)
            next_score = tuple(score[i] + part_score[i] for i in range(4))

            used.add(output_mesh)
            chosen.append(output_mesh)
            search(position + 1, used, output_mesh, chosen, next_score)
            chosen.pop()
            used.remove(output_mesh)

    search(0, set(), -1, [], (0, 0, 0, 0))

    if not solutions:
        detail = '; '.join(
            f'{row["reported_output_mesh"]}->{matches}'
            for row, matches in zip(ordered, candidates)
        )
        raise ValueError(
            'Native body graft found compatible body meshes but could not produce a unique '
            f'order-preserving reconstructed mapping. Candidates: {detail}'
        )

    solutions.sort(key=lambda item: item[0], reverse=True)
    best_score = solutions[0][0]
    best = [item for item in solutions if item[0] == best_score]

    if len(best) != 1:
        detail = '; '.join(
            f'{row["reported_output_mesh"]}->{matches}'
            for row, matches in zip(ordered, candidates)
        )
        raise ValueError(
            'Native body graft reconstructed mesh mapping is ambiguous even after topology, '
            f'material, weighted-bone and ordering checks. Candidates: {detail}'
        )

    resolved_indices = best[0][1]
    resolved: list[dict[str, Any]] = []
    remap_details: list[dict[str, Any]] = []

    for row, output_mesh in zip(ordered, resolved_indices):
        updated = dict(row)
        updated['output_mesh'] = int(output_mesh)
        resolved.append(updated)

        remap_details.append({
            'slot': row['slot'],
            'native_mesh': int(row['native_mesh']),
            'reported_output_mesh': int(row['reported_output_mesh']),
            'resolved_output_mesh': int(output_mesh),
            'compacted': int(row['reported_output_mesh']) != int(output_mesh),
        })

    return resolved, remap_details



def _decode_attribute(data: bytes | bytearray, base: int, stride: int, offset: int, typ: int, n: int):
    specs = {
        0: ('<f4', 1), 1: ('<f4', 2), 2: ('<f4', 3), 3: ('<f4', 4),
        5: ('u1', 4), 6: ('<i2', 2), 7: ('<i2', 4), 8: ('u1', 4),
        9: ('<i2', 2), 10: ('<i2', 4), 13: ('<f2', 2), 14: ('<f2', 4), 17: ('u1', 8),
    }
    if typ not in specs:
        raise ValueError(('unsupported vertex type', typ))
    import numpy as np
    dtype, nc = specs[typ]
    dt = np.dtype(dtype)
    arr = np.ndarray((n, nc), dtype=dt, buffer=data, offset=base + offset, strides=(stride, dt.itemsize)).copy()
    if typ == 8:
        return arr.astype(np.float32) / 255.0
    if typ in (9, 10):
        return np.maximum(arr.astype(np.float32) / 32767.0, -1.0)
    if typ in (13, 14):
        return arr.astype(np.float32)
    return arr


def _parse_decl_elements(declaration: bytes) -> list[tuple[int, int, int, int, int]]:
    elems = []
    q = 0
    lim = min(len(declaration), 136)
    while q < lim:
        block = declaration[q]
        if block == 255:
            break
        doff, typ, usage, count = declaration[q + 1:q + 5]
        elems.append((block, doff, typ, usage, count))
        q += 8
    return elems


def _weighted_local_bone_ordinals_for_mesh(model: dict[str, Any], mesh_index: int) -> set[int]:
    import numpy as np
    info = model['infos'][0][mesh_index]
    decls = _parse_decl_elements(model['declarations'][mesh_index])
    decl_by_usage: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for d in decls:
        decl_by_usage.setdefault(d[3], []).append(d)
    bid = decl_by_usage.get(2, [None])[0]
    bwd = decl_by_usage.get(1, [None])[0]
    if bid is None or bwd is None:
        return set()

    def attr(d):
        block, doff, typ, usage, cnt = d
        if block > 2 or info['stride'][block] <= 0:
            raise ValueError(('bad stream', block, info['stride']))
        return _decode_attribute(model['data'], model['lods'][0]['voff'] + info['vdo'][block], info['stride'][block], doff, typ, int(info['vcount']))

    local_joints = np.asarray(attr(bid), dtype=np.int64)
    weights = np.asarray(attr(bwd), dtype=np.float32)
    if local_joints.ndim != 2 or weights.ndim != 2:
        return set()
    k = min(local_joints.shape[1], weights.shape[1])
    local_joints = local_joints[:, :k]
    weights = weights[:, :k]
    bone_set_index = int(info['bone_set'])
    if bone_set_index < 0 or bone_set_index >= len(model['bone_sets']):
        return set()
    bone_set_count = len(model['bone_sets'][bone_set_index])
    valid = (local_joints >= 0) & (local_joints < bone_set_count) & (weights > 1e-8)
    return {int(value) for value in local_joints[valid].tolist()} if np.any(valid) else set()


def _weighted_bone_names_for_mesh(model: dict[str, Any], mesh_index: int) -> set[str]:
    info = model['infos'][0][mesh_index]
    bone_set_index = int(info['bone_set'])
    if bone_set_index < 0 or bone_set_index >= len(model['bone_sets']):
        return set()
    bone_set = model['bone_sets'][bone_set_index]
    names: set[str] = set()
    for ordinal in _weighted_local_bone_ordinals_for_mesh(model, mesh_index):
        if ordinal < 0 or ordinal >= len(bone_set):
            continue
        global_index = int(bone_set[ordinal])
        if 0 <= global_index < len(model['joint_names']):
            names.add(model['joint_names'][global_index])
    return names

def graft_native_body(imported_mdl: str | Path, output_mdl: str | Path, conversion_report: str | Path, target_body_mdls: dict[str, str], source_model_mdl: str | Path | None = None) -> dict[str, Any]:
    imported_mdl = Path(imported_mdl).resolve()
    output_mdl = Path(output_mdl).resolve()
    conversion_report = Path(conversion_report).resolve()
    if not imported_mdl.exists(): raise FileNotFoundError(imported_mdl)
    if not conversion_report.exists(): raise FileNotFoundError(conversion_report)
    report = json.loads(conversion_report.read_text(encoding='utf-8'))
    mapping = _mapping_from_report(report, target_body_mdls)
    suppression_rows=list((((report.get('transplant') or {}).get('body_suppression') or {}).get('native_suppression') or []))
    suppression_by_native: dict[tuple[str,int], set[int]] = {}
    for row in suppression_rows:
        key=(str(row.get('slot') or ''),int(row.get('native_mesh',-1)));suppression_by_native.setdefault(key,set()).update(int(value) for value in (row.get('triangles') or []))

    output = _parse(imported_mdl)
    lod0 = output['lods'][0]
    for index, lod in enumerate(output['lods'][1:], start=1):
        if int(lod['std_c']) or int(lod['vsize']) or int(lod['isize']):
            raise NotImplementedError(f'Native body graft refuses a reconstructed model with populated LOD{index}; preserving extra LOD buffers is not implemented.')
    if lod0['std_i'] != 0:
        raise NotImplementedError('Native body graft currently requires LOD0 standard meshes to begin at index 0.')

    native_models: dict[str, dict[str, Any]] = {}
    for row in mapping:
        native_models.setdefault(row['native_mdl'], _parse(row['native_mdl']))

    # The solver report retains the original sparse glTF/XIV mesh identity
    # (for example mesh 0, mesh 5, mesh 6). Penumbra's glTF -> MDL importer
    # compacts those identities into a dense MDL mesh array. Resolve the body
    # rows against the actually reconstructed topology instead of assuming the
    # reported mesh number is still a direct MDL array index.
    mapping, output_mesh_remap = _resolve_reconstructed_body_mapping(
        output,
        mapping,
        native_models,
    )

    mapped_output_meshes = {row['output_mesh'] for row in mapping}

    # Check topology and counts before touching the MDL.
    pair_details = []
    expected_index_blocks: dict[int, bytes] = {}
    topology_modes: dict[int, str] = {}
    exporter_dropped_by_output: dict[int, list[dict[str, int]]] = {}
    part_attribute_masks: dict[int, list[int]] = {}
    bone_set_requirements: dict[int, list[str]] = {}
    bone_set_owners: dict[int, set[int]] = {}
    for mesh_index, info in enumerate(output['infos'][0]):
        bone_set_owners.setdefault(int(info['bone_set']), set()).add(mesh_index)
    for row in mapping:
        native = native_models[row['native_mdl']]
        om = row['output_mesh']; nm = row['native_mesh']
        if nm < 0 or nm >= len(native['infos'][0]):
            raise ValueError(f'Native target mesh {nm} is outside {row["native_mdl"]}.')
        oi = output['infos'][0][om]; ni = native['infos'][0][nm]
        if int(oi['vcount']) != int(ni['vcount']):
            raise ValueError(
                f'Native body graft topology mismatch for output mesh {om} <- {row["slot"]} native mesh {nm}: '
                f'vertices {oi["vcount"]}/{ni["vcount"]}.'
            )
        if int(oi['part_count']) != int(ni['part_count']):
            raise ValueError(
                f'Native body graft part-count mismatch for output mesh {om} <- native mesh {nm}: '
                f'{oi["part_count"]}/{ni["part_count"]}. Refusing to mix native topology with incompatible reconstructed submesh parts.'
            )

        output_parts = oi.get('parts') or []
        native_parts = ni.get('parts') or []
        output_mesh_index_base = int(oi['idxoff'])
        native_mesh_index_base = int(ni['idxoff'])
        raw_part_offsets = [int(part['index_offset']) - native_mesh_index_base for part in native_parts]
        raw_part_counts = [int(part['index_count']) for part in native_parts]
        output_part_offsets = [int(part['index_offset']) - output_mesh_index_base for part in output_parts]
        output_part_counts = [int(part['index_count']) for part in output_parts]

        topology_mode = 'native'
        topology_block = _index_block(native, nm)
        topology_part_offsets = raw_part_offsets
        topology_part_counts = raw_part_counts
        exporter_dropped: list[dict[str, int]] = []
        raw_matches = (
            int(oi['icount']) == int(ni['icount'])
            and output_part_offsets == raw_part_offsets
            and output_part_counts == raw_part_counts
        )
        if not raw_matches:
            normalised = _penumbra_normalised_index_topology(native, nm)
            normalised_count = len(normalised['block']) // 2
            normalised_matches = (
                int(oi['icount']) == normalised_count
                and output_part_offsets == normalised['part_offsets']
                and output_part_counts == normalised['part_counts']
                and bool(normalised['dropped'])
            )
            if not normalised_matches:
                raise ValueError(
                    f'Native body graft topology mismatch for output mesh {om} <- {row["slot"]} native mesh {nm}: '
                    f'vertices {oi["vcount"]}/{ni["vcount"]}, indices {oi["icount"]}/{ni["icount"]}; '
                    f'Penumbra-normalised native indices={normalised_count}. The difference is not explained solely by '
                    'zero-area native triangles omitted by the Penumbra glTF exporter.'
                )
            topology_mode = 'penumbra-zero-area-normalised'
            topology_block = normalised['block']
            topology_part_offsets = normalised['part_offsets']
            topology_part_counts = normalised['part_counts']
            exporter_dropped = list(normalised['dropped'])

        for part_ordinal, (output_part, native_part) in enumerate(zip(output_parts, native_parts)):
            output_relative = output_part_offsets[part_ordinal]
            output_count = output_part_counts[part_ordinal]
            native_relative = raw_part_offsets[part_ordinal]
            native_count = raw_part_counts[part_ordinal]
            if output_relative < 0 or output_relative + output_count > int(oi['icount']):
                raise ValueError(
                    f'Reconstructed MeshPart range is outside output mesh {om}, part {part_ordinal}: '
                    f'relative offset/count {output_relative}/{output_count} within {oi["icount"]} indices.'
                )
            if native_relative < 0 or native_relative + native_count > int(ni['icount']):
                raise ValueError(
                    f'Native MeshPart range is outside target mesh {nm}, part {part_ordinal}: '
                    f'relative offset/count {native_relative}/{native_count} within {ni["icount"]} indices.'
                )
            if output_relative != topology_part_offsets[part_ordinal] or output_count != topology_part_counts[part_ordinal]:
                raise ValueError(
                    f'Native body graft MeshPart topology mismatch for output mesh {om}, part {part_ordinal}: '
                    f'relative offset/count {output_relative}/{output_count} reconstructed vs '
                    f'{topology_part_offsets[part_ordinal]}/{topology_part_counts[part_ordinal]} canonical native.'
                )
        suppressed = suppression_by_native.get((row['slot'], nm), set())
        expected_index_block = _suppress_index_block(topology_block, nm, suppressed) if suppressed else topology_block
        if len(expected_index_block) != int(oi['icount']) * 2:
            raise ValueError(
                f'Canonical native index payload size mismatch for output mesh {om}: '
                f'{len(expected_index_block) // 2}/{oi["icount"]} indices.'
            )
        expected_index_blocks[om] = expected_index_block
        topology_modes[om] = topology_mode
        exporter_dropped_by_output[om] = exporter_dropped
        output_attr = {name: index for index, name in enumerate(output['attributes'])}
        remapped_masks: list[int] = []
        for native_part in native_parts:
            native_mask = int(native_part['attribute_mask'])
            remapped = 0
            for bit in range(32):
                if not (native_mask & (1 << bit)):
                    continue
                if bit >= len(native['attributes']):
                    raise ValueError(f'Native body mesh {nm} uses unknown attribute bit {bit} in {row["native_mdl"]}.')
                attribute_name = native['attributes'][bit]
                output_bit = output_attr.get(attribute_name)
                if output_bit is None or output_bit >= 32:
                    raise ValueError(
                        f'Reconstructed model is missing native body attribute {attribute_name!r} required by '
                        f'{row["slot"]} mesh {nm}; refusing to break target-body option semantics.'
                    )
                remapped |= 1 << output_bit
            remapped_masks.append(remapped)
        part_attribute_masks[om] = remapped_masks
        native_bones = _bone_names(native, nm)
        output_bone_set = int(oi['bone_set'])
        existing = bone_set_requirements.get(output_bone_set)
        if existing is not None and existing != native_bones:
            raise ValueError(f'Two native target meshes require different local bone orders in shared output bone set {output_bone_set}.')
        owners = bone_set_owners.get(output_bone_set, set())
        foreign_owners = sorted(owners - mapped_output_meshes)
        if foreign_owners:
            current_bones = _bone_names(output, om)
            if current_bones != native_bones:
                raise ValueError(
                    f'Output body mesh {om} shares bone set {output_bone_set} with non-body mesh(es) {foreign_owners}; '
                    'the native local bone order differs, so grafting would alter garment skinning.'
                )
        bone_set_requirements[output_bone_set] = native_bones
        pair_details.append({
            **row,
            'vertices': int(oi['vcount']), 'indices': int(oi['icount']), 'parts': int(oi['part_count']),
            'native_strides': list(ni['stride']), 'reconstructed_strides': list(oi['stride']),
            'native_attribute_masks': [int(p['attribute_mask']) for p in native_parts],
            'restored_attribute_masks': remapped_masks,
            'native_bones': native_bones,
            'suppressed_triangles': len(suppression_by_native.get((row['slot'],nm),set())),
            'topology_mode': topology_modes[om],
            'exporter_zero_area_triangles_omitted': len(exporter_dropped_by_output[om]),
        })

    old = bytes(output['data'])
    prefix = bytearray(old[:lod0['voff']])
    old_index_buffer = bytearray(old[lod0['ioff']:lod0['ioff'] + lod0['isize']])
    suffix = old[lod0['ioff'] + lod0['isize']:]
    mapping_by_output = {row['output_mesh']: row for row in mapping}

    # Restore the exact native index payload into the rebuilt layout.
    for om, row in mapping_by_output.items():
        native = native_models[row['native_mdl']]; nm = row['native_mesh']
        oi = output['infos'][0][om]
        block = expected_index_blocks[om]
        start = int(oi['idxoff']) * 2
        old_index_buffer[start:start + len(block)] = block

    # Rebuild LOD0 body streams only; garment streams stay byte-for-byte intact.
    new_vertex_buffer = bytearray()
    new_vdo: dict[int, list[int]] = {}
    new_stride: dict[int, list[int]] = {}
    untouched_hashes = {}
    for mesh_index, info in enumerate(output['infos'][0]):
        new_vdo[mesh_index] = [0, 0, 0]
        new_stride[mesh_index] = list(info['stride'])
        row = mapping_by_output.get(mesh_index)
        if row is not None:
            native = native_models[row['native_mdl']]; ni = native['infos'][0][row['native_mesh']]
            for stream in range(3):
                stride = int(ni['stride'][stream])
                if stride <= 0: continue
                new_vdo[mesh_index][stream] = len(new_vertex_buffer)
                new_vertex_buffer.extend(_stream_block(native, row['native_mesh'], stream))
            new_stride[mesh_index] = list(ni['stride'])
        else:
            before = []
            for stream in range(3):
                stride = int(info['stride'][stream])
                if stride <= 0: continue
                new_vdo[mesh_index][stream] = len(new_vertex_buffer)
                block = _stream_block(output, mesh_index, stream)
                before.append(_sha(block))
                new_vertex_buffer.extend(block)
            untouched_hashes[mesh_index] = before

    # Restore native body vertex declarations, including extra UV/colour/tangent channels.
    for om, row in mapping_by_output.items():
        native = native_models[row['native_mdl']]
        dst = 68 + om * 136
        src = 68 + row['native_mesh'] * 136
        prefix[dst:dst + 136] = bytes(native['data'][src:src + 136])

    # Restore body MeshPart visibility attributes without touching garment parts.
    for om, masks in part_attribute_masks.items():
        for part, mask in zip(output['infos'][0][om]['parts'], masks):
            struct.pack_into('<I', prefix, int(part['struct_off']) + 8, int(mask))

    # Keep the importer's final buffer placement and only fix the body mesh offsets/strides.
    for mesh_index, info in enumerate(output['infos'][0]):
        struct_offset = int(info['struct_off'])
        struct.pack_into('<3i', prefix, struct_offset + 20, *new_vdo[mesh_index])
        struct.pack_into('<3B', prefix, struct_offset + 32, *new_stride[mesh_index])

    # XIV vertex BONE_INDEX values are UBYTE ordinals into a per-mesh bone set. The bone-set table itself
    # stores global joint indices. Preserve every local ordinal that carries positive weight exactly; an
    # importer may legitimately omit a global joint name that only occupied an unused local slot.
    output_global = {name: index for index, name in enumerate(output['joint_names'])}
    if not output_global:
        raise ValueError('Reconstructed model contains no global joints for native body grafting.')
    fallback_global_index = output_global.get('j_kosi', 0)
    weighted_ordinals_by_bone_set: dict[int, set[int]] = {}
    for row in mapping:
        native = native_models[row['native_mdl']]
        om = row['output_mesh']; nm = row['native_mesh']
        bone_set_index = int(output['infos'][0][om]['bone_set'])
        weighted_ordinals_by_bone_set.setdefault(bone_set_index, set()).update(_weighted_local_bone_ordinals_for_mesh(native, nm))

    substituted_unused_slots: dict[int, list[tuple[int, str]]] = {}
    for bone_set_index, native_names in bone_set_requirements.items():
        weighted_ordinals = weighted_ordinals_by_bone_set.get(bone_set_index, set())
        translated = []
        substitutions = []
        for ordinal, name in enumerate(native_names):
            global_index = output_global.get(name)
            if global_index is None:
                if ordinal in weighted_ordinals:
                    raise ValueError(
                        f'Reconstructed model lacks weighted native body bone {name!r} required by '
                        f'bone set {bone_set_index} local slot {ordinal}.'
                    )
                global_index = fallback_global_index
                substitutions.append((ordinal, name))
            translated.append(global_index)
        meta = output['bone_set_meta'][bone_set_index]
        if int(meta['count']) != len(translated):
            raise ValueError(
                f'Native body graft bone-set size mismatch for output set {bone_set_index}: '
                f'{meta["count"]} reconstructed vs {len(translated)} native.'
            )
        for ordinal, global_index in enumerate(translated):
            if global_index < -32768 or global_index > 32767:
                raise ValueError(f'Global joint index {global_index} exceeds XIV MDL bone-set int16 storage.')
            struct.pack_into('<h', prefix, int(meta['data_off']) + ordinal * 2, global_index)
        substituted_unused_slots[bone_set_index] = substitutions

    new_vertex_size = len(new_vertex_buffer)
    new_index_offset = lod0['ioff'] + (new_vertex_size - lod0['vsize'])
    struct.pack_into('<I', prefix, 28, new_index_offset)  # file header indexBufferOffsets[0]
    struct.pack_into('<I', prefix, 40, new_vertex_size)   # file header vertexBufferSizes[0]
    struct.pack_into('<i', prefix, lod0['struct_off'] + 44, new_vertex_size)
    struct.pack_into('<i', prefix, lod0['struct_off'] + 56, new_index_offset)

    # Keep the source equipment-model flags; the generic importer resets some of them.
    preserved_flags = None
    if source_model_mdl is not None:
        source_model = _parse(Path(source_model_mdl).resolve())
        preserved_flags = {}
        for name, relative in (('Flags1', 19), ('Flags2', 23), ('Flags3', 36)):
            value = int(source_model['model_data'][name])
            prefix[output['model_data_offset'] + relative] = value & 0xFF
            preserved_flags[name] = value

    result = bytes(prefix) + bytes(new_vertex_buffer) + bytes(old_index_buffer) + suffix
    output_mdl.parent.mkdir(parents=True, exist_ok=True)
    output_mdl.write_bytes(result)

    # Final audit: native body data must match and non-body data must still be untouched.
    final = _parse(output_mdl)
    body_checks = {}
    for om, row in mapping_by_output.items():
        native = native_models[row['native_mdl']]; nm = row['native_mesh']
        stream_checks = []
        for stream in range(3):
            native_block = _stream_block(native, nm, stream)
            final_block = _stream_block(final, om, stream)
            stream_checks.append({
                'stream': stream, 'present': bool(native_block), 'exact': native_block == final_block,
                'native_sha256': _sha(native_block), 'output_sha256': _sha(final_block),
            })
        suppressed=suppression_by_native.get((row['slot'],nm),set())
        expected_index = expected_index_blocks[om]
        index_exact = _index_block(final, om) == expected_index
        final_bones = _bone_names(final, om)
        native_bones = _bone_names(native, nm)
        weighted_ordinals = _weighted_local_bone_ordinals_for_mesh(native, nm)
        weighted_bones_exact = all(
            ordinal < len(final_bones) and ordinal < len(native_bones) and final_bones[ordinal] == native_bones[ordinal]
            for ordinal in weighted_ordinals
        )
        full_bones_exact = final_bones == native_bones
        declaration_exact = final['declarations'][om] == native['declarations'][nm]
        final_masks = [int(p['attribute_mask']) for p in final['infos'][0][om]['parts']]
        expected_masks = part_attribute_masks[om]
        attribute_masks_exact = final_masks == expected_masks
        if not all(item['exact'] for item in stream_checks) or not index_exact or not weighted_bones_exact or not declaration_exact or not attribute_masks_exact:
            raise ValueError(f'Native body graft post-write audit failed for output mesh {om}.')
        bone_set_index = int(final['infos'][0][om]['bone_set'])
        body_checks[str(om)] = {
            'slot': row['slot'], 'native_mesh': nm, 'streams': stream_checks,
            'indices_exact': index_exact, 'weighted_bone_names_exact': weighted_bones_exact,
            'full_bone_names_exact': full_bones_exact,
            'unused_bone_substitutions': [
                {'local_slot': ordinal, 'native_name': name}
                for ordinal, name in substituted_unused_slots.get(bone_set_index, [])
            ],
            'declaration_exact': declaration_exact,
            'attribute_masks_exact': attribute_masks_exact, 'attribute_masks': final_masks,
            'suppressed_triangles': len(suppressed),
            'topology_mode': topology_modes[om],
            'exporter_zero_area_triangles_omitted': len(exporter_dropped_by_output[om]),
        }

    garment_checks = {}
    for mesh_index, before_hashes in untouched_hashes.items():
        after_hashes = []
        for stream in range(3):
            if int(output['infos'][0][mesh_index]['stride'][stream]) <= 0: continue
            after_hashes.append(_sha(_stream_block(final, mesh_index, stream)))
        index_exact = _index_block(final, mesh_index) == _index_block(output, mesh_index)
        exact = before_hashes == after_hashes and index_exact
        if not exact:
            raise ValueError(f'Native body graft altered non-body output mesh {mesh_index}.')
        garment_checks[str(mesh_index)] = {'vertex_streams_exact': before_hashes == after_hashes, 'indices_exact': index_exact}

    if preserved_flags is not None:
        for key, expected in preserved_flags.items():
            if int(final['model_data'][key]) != expected:
                raise ValueError(f'Native body graft failed to preserve source model {key}: {final["model_data"][key]} != {expected}')

    return {
        'ok': True,
        'imported_mdl': str(imported_mdl), 'output_mdl': str(output_mdl), 'conversion_report': str(conversion_report),
        'input_sha256': _sha(imported_mdl.read_bytes()), 'output_sha256': _sha(result),
        'old_vertex_buffer_size': int(lod0['vsize']), 'new_vertex_buffer_size': int(new_vertex_size),
        'body_meshes': pair_details, 'body_checks': body_checks, 'non_body_checks': garment_checks,
        'output_mesh_remap': output_mesh_remap,
        'preserved_model_flags': preserved_flags,
    }
