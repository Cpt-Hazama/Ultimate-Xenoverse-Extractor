from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import struct


class BacFormatError(ValueError):
    pass


# Record sizes taken directly from Xv2CoreLib's BAC type readers.  We only
# decode event categories whose layouts/IDs are useful to Source right now.
_RECORD_SIZES = {
    0: 36, 1: 64, 2: 36, 3: 12, 4: 12, 5: 16, 6: 16, 7: 12,
    8: 48, 9: 64, 10: 76, 11: 16, 12: 12, 13: 12, 14: 12,
    15: 32, 16: 32, 17: 32, 18: 32, 19: 16, 20: 48, 21: 32,
    22: 48, 23: 64, 24: 56, 25: 16, 26: 80, 27: 24, 28: 36,
    29: 60, 30: 48, 31: 64,
}


@dataclass(slots=True, frozen=True)
class BacAnimation:
    start: int
    duration: int
    ean_type: int
    ean_index: int
    anim_flags: int
    start_frame: int
    end_frame: int
    loop_start_frame: int
    time_scale: float

    @property
    def end_time(self) -> int:
        # Duration 0 is used as an instantaneous/indefinite marker in a few
        # files. Treat it as active for the rest of the BAC entry.
        return self.start + self.duration if self.duration else 0x10000


@dataclass(slots=True, frozen=True)
class BacEvent:
    type_id: int
    start: int
    duration: int
    text: str


@dataclass(slots=True)
class BacEntry:
    index: int
    flags: int
    animations: list[BacAnimation] = field(default_factory=list)
    events: list[BacEvent] = field(default_factory=list)


@dataclass(slots=True)
class BacFile:
    path: Path
    entries: list[BacEntry]


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from('<H', data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from('<I', data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from('<i', data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from('<f', data, offset)[0]


def _require(data: bytes, offset: int, size: int, what: str) -> None:
    if offset < 0 or offset + size > len(data):
        raise BacFormatError(f'{what} points outside the BAC file')


def _decode_animation(raw: bytes) -> BacAnimation:
    # Exact BAC_Type0 layout from Xv2CoreLib/BAC/BAC_File.cs.
    time_scale = _f32(raw, 24)
    if not math.isfinite(time_scale) or time_scale <= 0.0:
        time_scale = 1.0
    return BacAnimation(
        start=_u16(raw, 0),
        duration=_u16(raw, 2),
        ean_type=_u16(raw, 8),
        ean_index=_u16(raw, 10),
        anim_flags=_u16(raw, 12),
        start_frame=_u16(raw, 16),
        end_frame=_u16(raw, 18),
        loop_start_frame=_u16(raw, 20),
        time_scale=time_scale,
    )


def _fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    rounded = round(value, 4)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:g}"


def _decode_event(type_id: int, raw: bytes) -> BacEvent | None:
    """Decode BAC timeline records using Xv2CoreLib's confirmed layouts.

    The event text intentionally preserves numeric IDs/flags. Their human-readable
    names depend on external game tables, but the values are still useful to the
    Source-side event handler and no longer disappear from the generated QCI.
    """
    start = _u16(raw, 0)
    duration = _u16(raw, 2)

    if type_id == 1:
        bdm_entry = _u16(raw, 8)
        bdm_file = raw[19] & 0x0F
        return BacEvent(type_id, start, duration, f'attack {bdm_file} {bdm_entry}')
    if type_id == 2:
        flags = _u16(raw, 8)
        x, y, z = _f32(raw, 12), _f32(raw, 16), _f32(raw, 20)
        return BacEvent(type_id, start, duration, f'movement {flags} {_fmt_float(x)} {_fmt_float(y)} {_fmt_float(z)}')
    if type_id == 3:
        return BacEvent(type_id, start, duration, f'invulnerable {_u16(raw, 8)}')
    if type_id == 4:
        return BacEvent(type_id, start, duration, f'timescale {_fmt_float(_f32(raw, 8))}')
    if type_id == 5:
        return BacEvent(type_id, start, duration, f'tracking {_fmt_float(_f32(raw, 8))} {_u16(raw, 12)}')
    if type_id == 6:
        return BacEvent(type_id, start, duration, f'charge {_u16(raw, 12)} {_i32(raw, 8)}')
    if type_id == 7:
        return BacEvent(type_id, start, duration, f'callback {_u32(raw, 8)}')
    if type_id == 8:
        eepk_type = _u16(raw, 8)
        skill_id = _u16(raw, 12)
        effect_id = _i32(raw, 16)
        text = f'fx {eepk_type} {effect_id}'
        if skill_id != 0xFFFF:
            text += f' {skill_id}'
        return BacEvent(type_id, start, duration, text)
    if type_id == 9:
        return BacEvent(type_id, start, duration, f'projectile {_u16(raw, 8)} {_i32(raw, 12)}')
    if type_id == 10:
        return BacEvent(type_id, start, duration, f'camera {_u16(raw, 8)} {_u16(raw, 12)} {_u16(raw, 14)}')
    if type_id == 11:
        acb_type = _u16(raw, 8)
        cue_id = _u16(raw, 12)
        prefix = 'vo' if acb_type in (3, 11) else 'se'
        return BacEvent(type_id, start, duration, f'{prefix} {acb_type} {cue_id}')
    if type_id == 12:
        return BacEvent(type_id, start, duration, f'targeting {_u16(raw, 8)} {_u16(raw, 10)}')
    if type_id == 13:
        return BacEvent(type_id, start, duration, f'part_visibility {_u16(raw, 8)} {_u16(raw, 10)}')
    if type_id == 14:
        return BacEvent(type_id, start, duration, f'anim_mod {_u16(raw, 8)}')
    if type_id == 15:
        params = ' '.join(_fmt_float(_f32(raw, off)) for off in (12, 16, 20, 24, 28))
        return BacEvent(type_id, start, duration, f'function {_i32(raw, 8)} {params}')
    if type_id == 16:
        return BacEvent(type_id, start, duration, f'postfx {_u16(raw, 8)} {_u16(raw, 12)} {_u16(raw, 14)}')
    if type_id == 17:
        return BacEvent(type_id, start, duration, f'throw {_u16(raw, 8)} {_u16(raw, 12)} {_u16(raw, 14)} {_u16(raw, 16)}')
    if type_id == 18:
        return BacEvent(type_id, start, duration, f'physics {_u16(raw, 8)} {_u16(raw, 12)}')
    if type_id == 19:
        return BacEvent(type_id, start, duration, f'aura {_u16(raw, 8)} {_u16(raw, 10)}')
    if type_id == 20:
        speed_flags = _u16(raw, 10)
        speed = _f32(raw, 12) if speed_flags & 0x8000 else float(_u32(raw, 12))
        return BacEvent(type_id, start, duration, f'homing {_u16(raw, 8)} {speed_flags} {_fmt_float(speed)}')
    if type_id == 21:
        return BacEvent(type_id, start, duration, f'eyes {_u16(raw, 12)} {_u16(raw, 14)}')
    if type_id == 22:
        return BacEvent(type_id, start, duration, f'bac_type22 {_u16(raw, 8)} {_u16(raw, 10)}')
    if type_id == 23:
        return BacEvent(type_id, start, duration, f'transparency {raw[8]}')
    if type_id == 24:
        return BacEvent(type_id, start, duration, f'dual_skill {_u16(raw, 8)}')
    if type_id == 25:
        return BacEvent(type_id, start, duration, f'charge_ex {_i32(raw, 12)} {_i32(raw, 8)}')
    if type_id == 26:
        return BacEvent(type_id, start, duration, f'camera_ex {_i32(raw, 8)} {_i32(raw, 12)}')
    if type_id == 27:
        return BacEvent(type_id, start, duration, f'fx_control {_u16(raw, 10)} {_u16(raw, 8)} {_u16(raw, 12)} {_u16(raw, 16)}')
    if type_id in (28, 29, 30, 31):
        return BacEvent(type_id, start, duration, f'bac_type{type_id} {_u16(raw, 8)} {_u16(raw, 10)}')
    return None


def load_bac(path: str | Path) -> BacFile:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 0x60 or data[:4] != b'#BAC':
        raise BacFormatError(f'{path.name} is not a valid XV2 BAC file')

    entry_count = _u32(data, 8)
    entry_offset = _u32(data, 16)
    _require(data, entry_offset, entry_count * 16, 'BAC entry table')

    # Type 17 exists in two layouts in retail BAC files: a full 32-byte
    # record and a compact 20-byte record. Xv2CoreLib determines which one is
    # present from the distance to the next data block. Build the same global
    # offset list once so giant/alternate-character BACs do not fail parsing.
    record_offsets: list[int] = []
    for scan_index in range(entry_count):
        scan_base = entry_offset + scan_index * 16
        scan_type_count = _u16(data, scan_base + 4)
        scan_type_list = _u32(data, scan_base + 8)
        if not scan_type_count or not scan_type_list:
            continue
        if scan_type_list < 0 or scan_type_list + scan_type_count * 16 > len(data):
            continue
        for scan_type_index in range(scan_type_count):
            scan_descriptor = scan_type_list + scan_type_index * 16
            scan_record_offset = _u32(data, scan_descriptor + 8)
            if scan_record_offset:
                record_offsets.append(scan_record_offset)
    record_offsets = sorted(set(record_offsets + [len(data)]))

    def next_record_offset(offset: int) -> int:
        for candidate in record_offsets:
            if candidate > offset:
                return candidate
        return len(data)

    entries: list[BacEntry] = []
    for entry_index in range(entry_count):
        base = entry_offset + entry_index * 16
        flags = _u32(data, base)
        if flags & 0x80000000:
            continue
        type_count = _u16(data, base + 4)
        type_list_offset = _u32(data, base + 8)
        if not type_count or not type_list_offset:
            continue
        _require(data, type_list_offset, type_count * 16, f'BAC entry {entry_index} type table')

        entry = BacEntry(index=entry_index, flags=flags)
        for type_index in range(type_count):
            descriptor = type_list_offset + type_index * 16
            type_id = _u16(data, descriptor)
            record_count = _u16(data, descriptor + 2)
            record_offset = _u32(data, descriptor + 8)
            record_size = _RECORD_SIZES.get(type_id)
            if type_id == 17 and record_count and record_offset:
                span = next_record_offset(record_offset) - record_offset
                candidate_size = span // record_count if span > 0 and span % record_count == 0 else 0
                if candidate_size in (20, 32):
                    record_size = candidate_size
            if not record_size or not record_count or not record_offset:
                continue
            _require(
                data,
                record_offset,
                record_count * record_size,
                f'BAC entry {entry_index} type {type_id}',
            )
            for record_index in range(record_count):
                offset = record_offset + record_index * record_size
                raw = data[offset:offset + record_size]
                if type_id == 0:
                    entry.animations.append(_decode_animation(raw))
                else:
                    event = _decode_event(type_id, raw)
                    if event is not None:
                        entry.events.append(event)

        if entry.animations or entry.events:
            entries.append(entry)

    return BacFile(path=path, entries=entries)


def _animation_for_event(animations: list[BacAnimation], event_time: int) -> BacAnimation | None:
    """Return the Type 0 command that owns ``event_time``.

    A Type 0 record's ``duration`` describes the command/playback segment and is
    not a reliable boundary for every later BAC record in the same action.
    Ownership therefore begins at a character animation command's start time and
    lasts until the next character animation command starts.
    """
    eligible = [animation for animation in animations if animation.start <= event_time]
    if not eligible:
        return None
    return max(eligible, key=lambda animation: animation.start)


def _map_event_frame(
    animation: BacAnimation,
    event_time: int,
    frame_count: int | None,
) -> int | None:
    """Map a BAC timeline time into the referenced EAN animation.

    Do not discard records merely because they fall beyond Type 0's duration or
    EndFrame. XV2 commonly leaves later hitbox/sound/effect records under the
    most recently started animation command. Looping ranges are wrapped; other
    out-of-range values are clamped to the exported SMD instead of disappearing.
    """
    local_time = max(0, event_time - animation.start)
    frame = int(round(animation.start_frame + local_time * animation.time_scale))

    if frame_count is None or frame_count <= 0:
        return max(0, frame)

    if 0 <= frame < frame_count:
        return frame

    loop_start = animation.loop_start_frame
    loop_end = animation.end_frame
    if (
        loop_start not in (0xFFFF,)
        and loop_end not in (0, 0xFFFF)
        and 0 <= loop_start < loop_end < frame_count
        and frame > loop_end
    ):
        loop_length = loop_end - loop_start + 1
        return loop_start + ((frame - loop_start) % loop_length)

    return min(max(frame, 0), frame_count - 1)


def animation_events(
    bac: BacFile,
    *,
    ean_type: int = 5,
    animation_frame_counts: dict[int, int] | None = None,
) -> dict[int, list[tuple[int, str]]]:
    """Map BAC records onto character EAN animation frames.

    Events are owned by the latest character Type 0 command that started at or
    before them, continuing until another character Type 0 command takes over.
    Deduplication is deliberately conservative: only records with the same EAN,
    mapped frame, BAC type, duration, and full payload are considered identical.
    """
    frame_counts = animation_frame_counts or {}
    mapped: dict[int, list[tuple[int, int, int, str]]] = {}
    seen: dict[int, set[tuple[int, int, int, str]]] = {}

    for entry in bac.entries:
        character_anims = sorted(
            (animation for animation in entry.animations if animation.ean_type == ean_type),
            key=lambda animation: animation.start,
        )
        if not character_anims:
            continue

        for event in entry.events:
            animation = _animation_for_event(character_anims, event.start)
            if animation is None:
                continue

            frame = _map_event_frame(
                animation,
                event.start,
                frame_counts.get(animation.ean_index),
            )
            if frame is None:
                continue

            key = (frame, event.type_id, event.duration, event.text)
            animation_seen = seen.setdefault(animation.ean_index, set())
            if key in animation_seen:
                continue
            animation_seen.add(key)
            mapped.setdefault(animation.ean_index, []).append(key)

    return {
        index: [(frame, text) for frame, _type_id, _duration, text in sorted(events, key=lambda item: (item[0], item[1], item[2], item[3]))]
        for index, events in mapped.items()
    }

