from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct


class EmdFormatError(ValueError):
    pass


@dataclass(slots=True)
class EmdTextureDefinition:
    flag0: int
    texture_index: int
    address_mode_uv: int
    filtering: int
    scale_u: float
    scale_v: float


@dataclass(slots=True)
class EmdTriangleGroup:
    indices: list[int] = field(default_factory=list)
    bone_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EmdSubmesh:
    name: str
    vertex_flags: int
    vertex_size: int
    positions: list[tuple[float, float, float]]
    normals: list[tuple[float, float, float] | None]
    uvs: list[tuple[float, float] | None]
    blend_indices: list[tuple[int, int, int, int] | None]
    blend_weights: list[tuple[float, float, float, float] | None]
    texture_definitions: list[EmdTextureDefinition]
    triangle_groups: list[EmdTriangleGroup]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]

    @property
    def triangle_count(self) -> int:
        return sum(len(group.indices) // 3 for group in self.triangle_groups)


@dataclass(slots=True)
class EmdMesh:
    name: str
    submeshes: list[EmdSubmesh]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]


@dataclass(slots=True)
class EmdModel:
    name: str
    meshes: list[EmdMesh]


@dataclass(slots=True)
class EmdFile:
    path: Path
    version: tuple[int, int, int, int]
    endian: str
    models: list[EmdModel]

    @property
    def vertex_count(self) -> int:
        return sum(len(s.positions) for m in self.models for mesh in m.meshes for s in mesh.submeshes)

    @property
    def triangle_count(self) -> int:
        return sum(s.triangle_count for m in self.models for mesh in m.meshes for s in mesh.submeshes)

    @property
    def submeshes(self) -> list[EmdSubmesh]:
        return [s for m in self.models for mesh in m.meshes for s in mesh.submeshes]


class _Reader:
    def __init__(self, data: bytes, endian: str) -> None:
        self.data = data
        self.endian = endian

    def unpack(self, fmt: str, offset: int):
        size = struct.calcsize(self.endian + fmt)
        if offset < 0 or offset + size > len(self.data):
            raise EmdFormatError(f"Read outside file at 0x{offset:X} (size 0x{size:X})")
        return struct.unpack_from(self.endian + fmt, self.data, offset)

    def u32(self, offset: int) -> int:
        return self.unpack("I", offset)[0]

    def cstring(self, offset: int) -> str:
        if offset < 0 or offset >= len(self.data):
            raise EmdFormatError(f"String offset outside file: 0x{offset:X}")
        end = self.data.find(b"\0", offset)
        if end < 0:
            end = len(self.data)
        return self.data[offset:end].decode("utf-8", errors="replace")


def _half(reader: _Reader, offset: int) -> float:
    return reader.unpack("e", offset)[0]


def _read_vertex(reader: _Reader, offset: int, flags: int):
    cursor = offset
    pos = (0.0, 0.0, 0.0)
    normal = None
    uv = None
    blend_indices = None
    blend_weights = None
    compressed = bool(flags & 0x8000)
    if flags & 0x1:
        pos = reader.unpack("3f", cursor); cursor += 12
    if flags & 0x2:
        if compressed:
            normal = (_half(reader, cursor), _half(reader, cursor + 2), _half(reader, cursor + 4)); cursor += 8
        else:
            normal = reader.unpack("3f", cursor); cursor += 12
    if flags & 0x4:
        if compressed:
            uv = (_half(reader, cursor), _half(reader, cursor + 2)); cursor += 4
        else:
            uv = reader.unpack("2f", cursor); cursor += 8
    if flags & 0x8:
        cursor += 4 if compressed else 8
    if flags & 0x80:
        cursor += 8 if compressed else 12
    if flags & 0x40:
        cursor += 4
    if flags & 0x200:
        raw_indices = reader.unpack("4B", cursor)
        cursor += 4
        if reader.endian == ">":
            raw_indices = tuple(reversed(raw_indices))
        if compressed:
            w0 = _half(reader, cursor)
            w1 = _half(reader, cursor + 2)
            w2 = _half(reader, cursor + 4)
            cursor += 8
        else:
            w0, w1, w2 = reader.unpack("3f", cursor)
            cursor += 12
        w3 = 1.0 - (w0 + w1 + w2)
        weights = [max(0.0, float(w0)), max(0.0, float(w1)), max(0.0, float(w2)), max(0.0, float(w3))]
        total = sum(weights)
        if total > 1e-8:
            weights = [w / total for w in weights]
        else:
            weights = [1.0, 0.0, 0.0, 0.0]
        blend_indices = tuple(int(v) for v in raw_indices)
        blend_weights = tuple(weights)
    return pos, normal, uv, blend_indices, blend_weights


def _parse_triangle_group(reader: _Reader, base: int) -> EmdTriangleGroup:
    face_count, bone_count, face_offset, _bone_offset = reader.unpack("IIII", base)
    if face_count > 20_000_000 or bone_count > 4096:
        raise EmdFormatError("Implausible triangle group counts")
    if not face_offset:
        face_offset = 0x10
    use_u32 = face_count > 0xFFFF
    stride = 4 if use_u32 else 2
    fmt = "I" if use_u32 else "H"
    indices = [reader.unpack(fmt, base + face_offset + i * stride)[0] for i in range(face_count)]
    names_table = face_offset + face_count * stride
    if names_table % 4 == 2:
        names_table += 2
    bone_names = [reader.cstring(base + reader.u32(base + names_table + i * 4)) for i in range(bone_count)]
    return EmdTriangleGroup(indices, bone_names)


def _parse_submesh(reader: _Reader, base: int) -> EmdSubmesh:
    bounds = reader.unpack("12f", base)
    flags, vertex_size, vertex_count, vertex_offset, name_offset = reader.unpack("IIIII", base + 0x30)
    _unknown, texture_count, triangle_group_count = reader.unpack("BBH", base + 0x44)
    texture_offset, triangles_offset = reader.unpack("II", base + 0x48)
    if vertex_count > 10_000_000 or triangle_group_count > 16384 or vertex_size < 12:
        raise EmdFormatError("Implausible submesh layout")

    definitions = []
    for i in range(texture_count):
        dbase = base + texture_offset + i * 12
        flag0, tex_index, address, filtering = reader.unpack("4B", dbase)
        scale_u, scale_v = reader.unpack("2f", dbase + 4)
        definitions.append(EmdTextureDefinition(flag0, tex_index, address, filtering, scale_u, scale_v))

    positions = []
    normals = []
    uvs = []
    blend_indices = []
    blend_weights = []
    for i in range(vertex_count):
        pos, normal, uv, indices, weights = _read_vertex(reader, base + vertex_offset + i * vertex_size, flags)
        positions.append(pos)
        normals.append(normal)
        uvs.append(uv)
        blend_indices.append(indices)
        blend_weights.append(weights)

    groups = []
    for i in range(triangle_group_count):
        relative = reader.u32(base + triangles_offset + i * 4)
        group_base = base + relative if relative else base + triangles_offset + 4
        groups.append(_parse_triangle_group(reader, group_base))

    return EmdSubmesh(reader.cstring(base + name_offset) or "null", flags, vertex_size, positions, normals, uvs,
                      blend_indices, blend_weights, definitions, groups,
                      (bounds[4], bounds[5], bounds[6]), (bounds[8], bounds[9], bounds[10]))


def _parse_mesh(reader: _Reader, base: int) -> EmdMesh:
    bounds = reader.unpack("12f", base)
    name_offset = reader.u32(base + 0x30)
    _unknown, count = reader.unpack("HH", base + 0x34)
    table = reader.u32(base + 0x38)
    submeshes = [_parse_submesh(reader, base + reader.u32(base + table + i * 4)) for i in range(count)]
    return EmdMesh(reader.cstring(base + name_offset), submeshes, (bounds[4], bounds[5], bounds[6]), (bounds[8], bounds[9], bounds[10]))


def _parse_model(reader: _Reader, base: int, name: str) -> EmdModel:
    _unknown, count = reader.unpack("HH", base)
    table = reader.u32(base + 4)
    return EmdModel(name, [_parse_mesh(reader, base + reader.u32(base + table + i * 4)) for i in range(count)])


def load_emd(path: str | Path) -> EmdFile:
    path = Path(path); data = path.read_bytes()
    if len(data) < 0x1C or data[:4] != b"#EMD":
        raise EmdFormatError(f"{path.name} is not an EMD file")
    marker = data[4:6]
    endian = "<" if marker == b"\xFE\xFF" else ">" if marker == b"\xFF\xFE" else "<"
    reader = _Reader(data, endian); version = tuple(data[8:12])
    _unknown, count = reader.unpack("HH", 0x10)
    model_table, names_table = reader.unpack("II", 0x14)
    names = [reader.cstring(reader.u32(names_table + i * 4)) for i in range(count)]
    models = [_parse_model(reader, reader.u32(model_table + i * 4), names[i]) for i in range(count)]
    return EmdFile(path, version, endian, models)
