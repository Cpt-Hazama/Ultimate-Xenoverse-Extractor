from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import bisect
import math
import struct


class EanFormatError(ValueError):
    pass


@dataclass(slots=True)
class EanKeyframe:
    frame: int
    value: tuple[float, float, float, float]


@dataclass(slots=True)
class EanTrack:
    flag: int
    unknown: int
    keyframes: list[EanKeyframe] = field(default_factory=list)

    def sample(self, frame: float, normalize_quaternion: bool = False) -> tuple[float, float, float, float] | None:
        if not self.keyframes:
            return None
        frames = [key.frame for key in self.keyframes]
        upper = bisect.bisect_left(frames, frame)
        if upper <= 0:
            value = self.keyframes[0].value
        elif upper >= len(self.keyframes):
            value = self.keyframes[-1].value
        else:
            left = self.keyframes[upper - 1]
            right = self.keyframes[upper]
            span = right.frame - left.frame
            t = 0.0 if span <= 0 else (frame - left.frame) / span
            value = tuple(left.value[i] + (right.value[i] - left.value[i]) * t for i in range(4))
        if normalize_quaternion:
            length = math.sqrt(sum(component * component for component in value))
            if length > 1e-8:
                value = tuple(component / length for component in value)
        return value


@dataclass(slots=True)
class EanNode:
    bone_index: int
    tracks: dict[int, EanTrack]
    bone_name: str | None = None


@dataclass(slots=True)
class EanAnimation:
    index: int
    name: str
    duration: int
    frame_index_size: int
    frame_float_size: int
    nodes: list[EanNode]

    @property
    def node_map(self) -> dict[int, EanNode]:
        return {node.bone_index: node for node in self.nodes}

    @property
    def node_name_map(self) -> dict[str, EanNode]:
        return {node.bone_name: node for node in self.nodes if node.bone_name}


@dataclass(slots=True)
class EanFile:
    path: Path
    version: tuple[int, int, int, int]
    endian: str
    animation_type: int
    animations: list[EanAnimation]
    skeleton_bone_names: list[str] = field(default_factory=list)

    @property
    def is_camera(self) -> bool:
        return self.animation_type == 1


class _Reader:
    def __init__(self, data: bytes, endian: str) -> None:
        self.data = data
        self.endian = endian

    def unpack(self, fmt: str, offset: int):
        size = struct.calcsize(self.endian + fmt)
        if offset < 0 or offset + size > len(self.data):
            raise EanFormatError(f"Read outside file at 0x{offset:X} (size 0x{size:X})")
        return struct.unpack_from(self.endian + fmt, self.data, offset)

    def u32(self, offset: int) -> int:
        return self.unpack("I", offset)[0]

    def cstring(self, offset: int) -> str:
        if offset < 0 or offset >= len(self.data):
            return ""
        end = self.data.find(b"\0", offset)
        if end < 0:
            end = len(self.data)
        return self.data[offset:end].decode("utf-8", errors="replace")


def _read_track(reader: _Reader, base: int, index_size: int, float_size: int) -> EanTrack:
    flag, unknown, _padding, count, frames_offset, values_offset = reader.unpack("BBHIII", base)
    if count > 2_000_000:
        raise EanFormatError(f"Implausible EAN keyframe count: {count}")
    frame_stride = 1 if index_size == 0 else 2 if index_size == 1 else 0
    value_fmt = "4e" if float_size == 1 else "4f" if float_size in (0, 2) else ""
    if not frame_stride or not value_fmt:
        raise EanFormatError(f"Unsupported EAN index/float size: {index_size}/{float_size}")
    keyframes: list[EanKeyframe] = []
    value_stride = struct.calcsize(reader.endian + value_fmt)
    for i in range(count):
        frame = reader.unpack("B" if frame_stride == 1 else "H", base + frames_offset + i * frame_stride)[0]
        value = tuple(float(v) for v in reader.unpack(value_fmt, base + values_offset + i * value_stride))
        keyframes.append(EanKeyframe(frame, value))
    keyframes.sort(key=lambda key: key.frame)
    return EanTrack(flag, unknown, keyframes)


def _read_node(reader: _Reader, base: int, index_size: int, float_size: int, bone_names: list[str]) -> EanNode:
    bone_index, track_count, table_offset = reader.unpack("HHI", base)
    if track_count > 64:
        raise EanFormatError(f"Implausible EAN track count: {track_count}")
    tracks: dict[int, EanTrack] = {}
    for i in range(track_count):
        relative = reader.u32(base + table_offset + i * 4)
        track = _read_track(reader, base + relative, index_size, float_size)
        tracks[track.flag] = track
    bone_name = bone_names[bone_index] if 0 <= bone_index < len(bone_names) else None
    return EanNode(bone_index, tracks, bone_name)


def _read_animation(reader: _Reader, base: int, index: int, name: str, bone_names: list[str]) -> EanAnimation:
    _unknown, index_size, float_size, duration, node_count, nodes_offset = reader.unpack("HBBIII", base)
    if duration > 10_000_000 or node_count > 100_000:
        raise EanFormatError("Implausible EAN animation layout")
    nodes: list[EanNode] = []
    for i in range(node_count):
        relative = reader.u32(base + nodes_offset + i * 4)
        nodes.append(_read_node(reader, base + relative, index_size, float_size, bone_names))
    return EanAnimation(index, name or f"Animation_{index:03d}", duration, index_size, float_size, nodes)



def _read_embedded_skeleton_names(reader: _Reader, base: int) -> list[str]:
    """Read the ESK skeleton section embedded directly inside an EAN file.

    EAN animation node indices reference this embedded skeleton, not necessarily
    the external character ESK's numeric bone order. Bone-name remapping is
    therefore required when applying an EAN to the loaded model skeleton.
    """
    if base <= 0 or base + 0x24 > len(reader.data):
        return []
    try:
        bone_count, _flags, _hierarchy_offset, names_offset, *_rest = reader.unpack(
            "HHIIIIIIII", base
        )
        if bone_count > 10000 or names_offset == 0:
            return []
        names: list[str] = []
        for index in range(bone_count):
            relative = reader.u32(base + names_offset + index * 4)
            names.append(reader.cstring(base + relative))
        return names
    except (EanFormatError, struct.error):
        return []

def load_ean(path: str | Path) -> EanFile:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 0x20 or data[:4] != b"#EAN":
        raise EanFormatError(f"{path.name} is not an EAN file")
    marker = data[4:6]
    endian = "<" if marker == b"\xFE\xFF" else ">" if marker == b"\xFF\xFE" else "<"
    reader = _Reader(data, endian)
    version = tuple(data[8:12])
    animation_type = data[0x10]
    animation_count = reader.unpack("H", 0x12)[0]
    skeleton_offset = reader.u32(0x14)
    animations_offset = reader.u32(0x18)
    names_offset = reader.u32(0x1C)
    if animation_count > 100_000:
        raise EanFormatError(f"Implausible EAN animation count: {animation_count}")
    bone_names = _read_embedded_skeleton_names(reader, skeleton_offset)
    names: list[str] = []
    for i in range(animation_count):
        name_address = reader.u32(names_offset + i * 4)
        names.append(reader.cstring(name_address))
    animations = []
    for i in range(animation_count):
        animation_address = reader.u32(animations_offset + i * 4)
        animations.append(_read_animation(reader, animation_address, i, names[i], bone_names))
    return EanFile(path, version, endian, animation_type, animations, bone_names)
