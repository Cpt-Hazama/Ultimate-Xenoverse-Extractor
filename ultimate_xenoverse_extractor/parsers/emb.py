from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import struct

from PIL import Image
from PySide6.QtGui import QImage


class EmbFormatError(ValueError):
    pass


@dataclass(slots=True)
class EmbEntry:
    index: int
    name: str
    data: bytes

    def to_qimage(self) -> QImage | None:
        try:
            with Image.open(BytesIO(self.data)) as image:
                rgba = image.convert("RGBA")
                raw = rgba.tobytes("raw", "RGBA")
                # copy() detaches the QImage from the temporary Python buffer.
                return QImage(raw, rgba.width, rgba.height, rgba.width * 4, QImage.Format_RGBA8888).copy()
        except Exception:
            return None


@dataclass(slots=True)
class EmbFile:
    path: Path
    version: tuple[int, int, int, int]
    endian: str
    entries: list[EmbEntry]

    @property
    def texture_count(self) -> int:
        return sum(entry.to_qimage() is not None for entry in self.entries)


class _Reader:
    def __init__(self, data: bytes, endian: str) -> None:
        self.data = data
        self.endian = endian

    def unpack(self, fmt: str, offset: int):
        size = struct.calcsize(self.endian + fmt)
        if offset < 0 or offset + size > len(self.data):
            raise EmbFormatError(f"Read outside file at 0x{offset:X}")
        return struct.unpack_from(self.endian + fmt, self.data, offset)

    def u32(self, offset: int) -> int:
        return self.unpack("I", offset)[0]

    def cstring(self, offset: int) -> str:
        if not 0 <= offset < len(self.data):
            return ""
        end = self.data.find(b"\0", offset)
        if end < 0:
            end = len(self.data)
        return self.data[offset:end].decode("utf-8", errors="replace")


def load_emb(path: str | Path) -> EmbFile:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 0x20 or data[:4] != b"#EMB":
        raise EmbFormatError(f"{path.name} is not an EMB file")

    marker = data[4:6]
    endian = "<" if marker == b"\xFE\xFF" else ">" if marker == b"\xFF\xFE" else "<"
    reader = _Reader(data, endian)
    version = tuple(data[8:12])
    file_count = reader.u32(0x0C)
    data_table = reader.u32(0x18)
    name_table = reader.u32(0x1C)
    if file_count > 100_000:
        raise EmbFormatError(f"Implausible EMB entry count: {file_count}")

    entries: list[EmbEntry] = []
    # EMB data offsets are relative to 0x20 + the current table-entry offset.
    for index in range(file_count):
        relative, size = reader.unpack("II", data_table + index * 8)
        absolute = 0x20 + index * 8 + relative
        if size and (absolute < 0 or absolute + size > len(data)):
            raise EmbFormatError(f"Entry {index} points outside the file")
        name = ""
        if name_table:
            name_offset = reader.u32(name_table + index * 4)
            name = reader.cstring(name_offset)
        if not name:
            name = f"DATA{index:03d}.dds"
        entries.append(EmbEntry(index, name, data[absolute:absolute + size] if size else b""))

    return EmbFile(path, version, endian, entries)
