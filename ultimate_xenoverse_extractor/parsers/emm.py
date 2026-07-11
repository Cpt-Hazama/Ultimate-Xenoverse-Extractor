from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct


class EmmFormatError(ValueError):
    pass


@dataclass(slots=True)
class EmmParameter:
    name: str
    type: int
    level: int
    value: float | int

    @property
    def type_name(self) -> str:
        return "Float" if self.type == 0 else "UInt" if self.type == 1 else f"Unknown (0x{self.type:X})"


@dataclass(slots=True)
class EmmMaterial:
    name: str
    shader: str
    unknown: int
    parameters: list[EmmParameter] = field(default_factory=list)

    def parameter(self, name: str) -> EmmParameter | None:
        wanted = name.casefold()
        return next((parameter for parameter in self.parameters if parameter.name.casefold() == wanted), None)

    def value(self, name: str, default: float | int = 0.0) -> float | int:
        parameter = self.parameter(name)
        return parameter.value if parameter is not None else default

    def color(self, slot: int = 2) -> tuple[float, float, float, float] | None:
        prefix = f"MatCol{slot}"
        names = [f"{prefix}{channel}" for channel in "RGBA"]
        if not any(self.parameter(name) for name in names):
            return None
        values = [float(self.value(name, 1.0 if name.endswith("A") else 0.0)) for name in names]
        return tuple(max(0.0, min(1.0, value)) for value in values)  # type: ignore[return-value]

    def preview_color(self) -> tuple[float, float, float, float]:
        # MatCol2 is the most useful visible tint on common XV2 character shaders.
        # Fall back through the other slots, then to a neutral material color.
        for slot in (2, 0, 1, 3):
            color = self.color(slot)
            if color is not None and max(color[:3]) > 0.015:
                return color
        return 0.52, 0.58, 0.66, 1.0


@dataclass(slots=True)
class EmmFile:
    path: Path
    version: tuple[int, int, int, int]
    endian: str
    header_size: int
    materials: list[EmmMaterial]
    default_values: bytes = b""

    def material(self, name: str) -> EmmMaterial | None:
        wanted = name.casefold()
        return next((material for material in self.materials if material.name.casefold() == wanted), None)

    @property
    def parameter_count(self) -> int:
        return sum(len(material.parameters) for material in self.materials)


class _Reader:
    def __init__(self, data: bytes, endian: str) -> None:
        self.data = data
        self.endian = endian

    def unpack(self, fmt: str, offset: int):
        size = struct.calcsize(self.endian + fmt)
        if offset < 0 or offset + size > len(self.data):
            raise EmmFormatError(f"Read outside file at 0x{offset:X} (size 0x{size:X})")
        return struct.unpack_from(self.endian + fmt, self.data, offset)

    def u16(self, offset: int) -> int:
        return self.unpack("H", offset)[0]

    def u32(self, offset: int) -> int:
        return self.unpack("I", offset)[0]

    def fixed_string(self, offset: int, length: int) -> str:
        if offset < 0 or offset + length > len(self.data):
            raise EmmFormatError(f"String outside file at 0x{offset:X}")
        return self.data[offset:offset + length].split(b"\0", 1)[0].decode("utf-8", errors="replace")


def load_emm(path: str | Path) -> EmmFile:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 0x10 or data[:4] != b"#EMM":
        raise EmmFormatError(f"{path.name} is not an EMM file")

    marker = data[4:6]
    endian = "<" if marker == b"\xFE\xFF" else ">" if marker == b"\xFF\xFE" else "<"
    reader = _Reader(data, endian)
    header_size = reader.u16(6)
    version = tuple(data[8:12])
    material_base = reader.u32(0x0C)
    if material_base <= 0 or material_base + 4 > len(data):
        raise EmmFormatError(f"Invalid material section offset: 0x{material_base:X}")

    material_count = reader.u32(material_base)
    if material_count > 65535:
        raise EmmFormatError(f"Implausible material count: {material_count}")

    materials: list[EmmMaterial] = []
    for index in range(material_count):
        relative = reader.u32(material_base + 4 + index * 4)
        base = material_base + relative
        name = reader.fixed_string(base, 32)
        shader = reader.fixed_string(base + 0x20, 32)
        parameter_count, unknown = reader.unpack("HH", base + 0x40)
        if parameter_count > 8192:
            raise EmmFormatError(f"Implausible parameter count for {name}: {parameter_count}")

        parameters: list[EmmParameter] = []
        parameter_base = base + 0x44
        for parameter_index in range(parameter_count):
            offset = parameter_base + parameter_index * 0x28
            parameter_name = reader.fixed_string(offset, 32)
            parameter_type, level = reader.unpack("HH", offset + 0x20)
            if parameter_type == 0:
                value: float | int = reader.unpack("f", offset + 0x24)[0]
            else:
                value = reader.u32(offset + 0x24)
            parameters.append(EmmParameter(parameter_name, parameter_type, level, value))
        materials.append(EmmMaterial(name, shader, unknown, parameters))

    default_values = b""
    if header_size > 0x10 and len(data) >= 0x14:
        default_offset = reader.u32(0x10)
        if 0 < default_offset < len(data):
            default_values = data[default_offset:min(len(data), default_offset + 68)]

    return EmmFile(path, version, endian, header_size, materials, default_values)
