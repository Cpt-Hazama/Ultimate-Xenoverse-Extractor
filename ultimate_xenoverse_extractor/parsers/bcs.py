from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


class BcsFormatError(ValueError):
    pass


@dataclass(slots=True)
class BcsPhysics:
    name: str
    files: tuple[str, ...] = ()

    def referenced_stems(self) -> list[str]:
        result: list[str] = []
        for value in self.files:
            stem = _clean_reference(value)
            if stem and stem not in result:
                result.append(stem)
        return result


@dataclass(slots=True)
class BcsPart:
    slot: int
    model: int
    model2: int
    texture: int
    name: str
    files: tuple[str, str, str, str]
    physics: list[BcsPhysics] = field(default_factory=list)

    def default_stem(self, suffix: str) -> str:
        prefix = self.name.strip("\x00 ")
        return f"{prefix}_{self.model:03d}_{suffix}" if prefix else ""

    def referenced_stems(self, suffix: str) -> list[str]:
        """Return every model/material/texture stem referenced by this part.

        BCS files can override individual companion files through four optional
        strings. Keeping all non-empty references is safer than assuming a fixed
        meaning for each slot, and the caller only accepts references that exist.
        """
        result: list[str] = []
        default = self.default_stem(suffix)
        if default:
            result.append(default)
        for value in self.files:
            stem = _clean_reference(value)
            if stem and stem not in result:
                result.append(stem)
        for physics in self.physics:
            for stem in physics.referenced_stems():
                if stem not in result:
                    result.append(stem)
        return result


@dataclass(slots=True)
class BcsPartSet:
    index: int
    parts: list[BcsPart | None]

    def part(self, slot: int) -> BcsPart | None:
        return self.parts[slot] if 0 <= slot < len(self.parts) else None


@dataclass(slots=True)
class BcsFile:
    path: Path
    endian: str
    version_header_size: int
    part_sets: list[BcsPartSet | None]

    def get_part(self, set_index: int, slot: int) -> BcsPart | None:
        if not 0 <= set_index < len(self.part_sets):
            return None
        part_set = self.part_sets[set_index]
        return part_set.part(slot) if part_set is not None else None


def _clean_reference(value: str) -> str:
    value = value.replace("\\", "/").rsplit("/", 1)[-1].strip().strip("\x00")
    if not value:
        return ""
    lower = value.casefold()
    for suffix in (".dyt.emb", ".emd", ".emm", ".emb", ".esk", ".ean", ".fce"):
        if lower.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _read_cstring(data: bytes, base: int, relative: int) -> str:
    if relative == 0:
        return ""
    offset = base + relative
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("utf-8", errors="replace")


def _require(data: bytes, offset: int, size: int, label: str) -> None:
    if offset < 0 or offset + size > len(data):
        raise BcsFormatError(f"{label} points outside the BCS file (0x{offset:X}).")


def load_bcs(path: Path) -> BcsFile:
    data = path.read_bytes()
    if len(data) < 0x4C or data[:4] != b"#BCS":
        raise BcsFormatError(f"{path.name} is not a valid BCS file.")

    endian_marker = data[4:6]
    if endian_marker == b"\xFE\xFF":
        endian = "<"
    elif endian_marker == b"\xFF\xFE":
        endian = ">"
    else:
        raise BcsFormatError(f"Unsupported BCS endian marker {endian_marker.hex()}.")

    u16 = lambda off: struct.unpack_from(endian + "H", data, off)[0]
    u32 = lambda off: struct.unpack_from(endian + "I", data, off)[0]

    header_size = u16(0x06)
    num_sets = u16(0x0C)
    table_offset = u32(0x18)
    _require(data, table_offset, num_sets * 4, "Part-set table")

    sets: list[BcsPartSet | None] = [None] * num_sets
    for set_index in range(num_sets):
        set_offset = u32(table_offset + set_index * 4)
        if set_offset == 0:
            continue
        _require(data, set_offset, 0x48, f"Part set {set_index}")

        parts: list[BcsPart | None] = [None] * 10
        for slot in range(10):
            relative = u32(set_offset + 0x20 + slot * 4)
            if relative == 0:
                continue
            part_offset = set_offset + relative
            _require(data, part_offset, 0x58, f"Part set {set_index}, slot {slot}")

            model, model2, texture = struct.unpack_from(endian + "HHH", data, part_offset)
            num_colors = u16(part_offset + 0x12)
            raw_name = data[part_offset + 0x34 : part_offset + 0x38]
            name = raw_name.split(b"\0", 1)[0].decode("ascii", errors="replace")
            file_offsets = struct.unpack_from(endian + "4I", data, part_offset + 0x38)
            files = tuple(_read_cstring(data, part_offset, rel) for rel in file_offsets)

            num_physics = u16(part_offset + 0x4A)
            physics_offset = part_offset + 0x58 + num_colors * 4
            physics: list[BcsPhysics] = []
            for physics_index in range(num_physics):
                current = physics_offset + physics_index * 0x48
                _require(data, current, 0x48, f"Physics object {physics_index}")
                physics_name = data[current + 0x24 : current + 0x28].split(b"\0", 1)[0].decode(
                    "ascii", errors="replace"
                )
                refs = struct.unpack_from(endian + "6I", data, current + 0x28)
                physics.append(
                    BcsPhysics(
                        name=physics_name,
                        files=tuple(_read_cstring(data, current, rel) for rel in refs),
                    )
                )

            parts[slot] = BcsPart(
                slot=slot,
                model=model,
                model2=model2,
                texture=texture,
                name=name,
                files=files,  # type: ignore[arg-type]
                physics=physics,
            )

        sets[set_index] = BcsPartSet(index=set_index, parts=parts)

    return BcsFile(path=path, endian=endian, version_header_size=header_size, part_sets=sets)
