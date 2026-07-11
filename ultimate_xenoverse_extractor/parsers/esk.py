from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


class EskFormatError(ValueError):
    pass


@dataclass(slots=True)
class EskBone:
    index: int
    name: str
    parent_index: int
    child_index: int
    sibling_index: int
    ik_flag: int
    position: tuple[float, float, float, float]
    rotation: tuple[float, float, float, float]
    scale: tuple[float, float, float, float]
    absolute_matrix: tuple[float, ...] | None = None

    @property
    def parent(self) -> int | None:
        return None if self.parent_index == 0xFFFF else self.parent_index


@dataclass(slots=True)
class EskFile:
    path: Path
    version: tuple[int, int, int, int]
    endian: str
    flags: int
    skeleton_id: int
    bones: list[EskBone]

    @property
    def roots(self) -> list[EskBone]:
        return [bone for bone in self.bones if bone.parent is None]


class _Reader:
    def __init__(self, data: bytes, endian: str) -> None:
        self.data = data
        self.endian = endian

    def unpack(self, fmt: str, offset: int):
        size = struct.calcsize(self.endian + fmt)
        if offset < 0 or offset + size > len(self.data):
            raise EskFormatError(f"Read outside file at 0x{offset:X} (size 0x{size:X})")
        return struct.unpack_from(self.endian + fmt, self.data, offset)

    def cstring(self, offset: int) -> str:
        if offset < 0 or offset >= len(self.data):
            raise EskFormatError(f"String offset outside file: 0x{offset:X}")
        end = self.data.find(b"\0", offset)
        if end < 0:
            end = len(self.data)
        return self.data[offset:end].decode("utf-8", errors="replace")


def load_esk(path: str | Path) -> EskFile:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 0x20 or data[:4] != b"#ESK":
        raise EskFormatError(f"{path.name} is not an ESK file")

    marker = data[4:6]
    if marker == b"\xFE\xFF":
        endian = "<"
    elif marker == b"\xFF\xFE":
        endian = ">"
    else:
        # XV2 PC files are little endian; retain a useful fallback for malformed tools.
        endian = "<"

    reader = _Reader(data, endian)
    header_size = reader.unpack("H", 0x06)[0]
    if header_size < 0x1C:
        raise EskFormatError(f"Unsupported ESK header size 0x{header_size:X}")

    version = tuple(data[0x08:0x0C])
    skeleton_offset = reader.unpack("I", 0x10)[0]
    if skeleton_offset == 0 or skeleton_offset + 0x24 > len(data):
        raise EskFormatError("Invalid skeleton section offset")

    base = skeleton_offset
    (
        bone_count,
        flags,
        hierarchy_offset,
        names_offset,
        relative_offset,
        absolute_offset,
        _ik_offset,
        _extra_offset,
        skeleton_id_low,
        skeleton_id_high,
    ) = reader.unpack("HHIIIIIIII", base)

    if bone_count > 10000:
        raise EskFormatError(f"Implausible ESK bone count: {bone_count}")

    skeleton_id = skeleton_id_low | (skeleton_id_high << 32)
    bones: list[EskBone] = []
    for index in range(bone_count):
        hierarchy = reader.unpack("HHHH", base + hierarchy_offset + index * 8)
        name_rel = reader.unpack("I", base + names_offset + index * 4)[0]
        name = reader.cstring(base + name_rel)
        values = reader.unpack("12f", base + relative_offset + index * 0x30)
        matrix = None
        if absolute_offset:
            matrix = tuple(reader.unpack("16f", base + absolute_offset + index * 0x40))
        bones.append(
            EskBone(
                index=index,
                name=name,
                parent_index=hierarchy[0],
                child_index=hierarchy[1],
                sibling_index=hierarchy[2],
                ik_flag=hierarchy[3],
                position=tuple(values[0:4]),
                rotation=tuple(values[4:8]),
                scale=tuple(values[8:12]),
                absolute_matrix=matrix,
            )
        )

    return EskFile(path, version, endian, flags, skeleton_id, bones)
