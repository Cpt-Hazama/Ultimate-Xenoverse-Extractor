from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


class SoundToolError(RuntimeError):
    pass


@dataclass(slots=True)
class SoundStream:
    index: int
    name: str
    sample_rate: int | None = None
    channels: int | None = None
    samples: int | None = None
    title: str | None = None
    artist: str | None = None
    metadata_loaded: bool = False


@dataclass(slots=True)
class SoundBank:
    acb_path: Path
    awb_path: Path | None
    streams: list[SoundStream]
    vgmstream_path: Path


_STREAM_COUNT_PATTERNS = (
    re.compile(r"stream\s+count\s*:\s*(\d+)", re.I),
    re.compile(r"subsong\s+count\s*:\s*(\d+)", re.I),
    re.compile(r"streams?\s*:\s*(\d+)", re.I),
    re.compile(r"stream\s+(?:total|count)\s*:\s*(\d+)", re.I),
    re.compile(r"total\s+subsongs?\s*:\s*(\d+)", re.I),
)
_STREAM_NAME_PATTERNS = (
    re.compile(r"^\s*stream\s+name\s*:\s*(.+?)\s*$", re.I | re.M),
    re.compile(r"^\s*name\s*:\s*(.+?)\s*$", re.I | re.M),
)
_TITLE_PATTERNS = (
    re.compile(r"^\s*(?:tag\s+)?title\s*:\s*(.+?)\s*$", re.I | re.M),
)
_ARTIST_PATTERNS = (
    re.compile(r"^\s*(?:tag\s+)?artist\s*:\s*(.+?)\s*$", re.I | re.M),
)
_RATE_RE = re.compile(r"sample\s+rate\s*:\s*(\d+)", re.I)
_CHANNELS_RE = re.compile(r"channels?\s*:\s*(\d+)", re.I)
_SAMPLES_RE = re.compile(r"(?:num\s+)?samples?\s*:\s*(\d+)", re.I)


def find_vgmstream(project_root: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if project_root is not None:
        candidates.extend(
            [
                project_root / "ultimate_xenoverse_extractor" / "tools" / "vgmstream-cli.exe",
                project_root / "ultimate_xenoverse_extractor" / "tools" / "vgmstream-cli",
                project_root / "tools" / "vgmstream-cli.exe",
                project_root / "tools" / "vgmstream-cli",
                project_root / "bin" / "vgmstream-cli.exe",
                project_root / "bin" / "vgmstream-cli",
                project_root / "vgmstream-cli.exe",
                project_root / "vgmstream-cli",
            ]
        )
    env = os.environ.get("VGMSTREAM_CLI")
    if env:
        candidates.insert(0, Path(env))
    for name in ("vgmstream-cli.exe", "vgmstream-cli", "test.exe", "test"):
        located = shutil.which(name)
        if located:
            candidates.append(Path(located))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _run(cli: Path, args: list[str], timeout: int = 120) -> str:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.run(
        [str(cli), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=creationflags,
    )
    if process.returncode != 0:
        raise SoundToolError(process.stdout.strip() or f"vgmstream exited with code {process.returncode}")
    return process.stdout


def _metadata(cli: Path, acb_path: Path, stream_index: int = 1) -> str:
    # Current vgmstream-cli accepts -m for metadata and -s for subsong selection.
    return _run(cli, ["-m", "-s", str(stream_index), str(acb_path)], timeout=60)


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _display_name(stream_name: str | None, title: str | None, artist: str | None, index: int) -> str:
    title = title.strip() if title else None
    artist = artist.strip() if artist else None
    stream_name = stream_name.strip() if stream_name else None
    if title and artist and title.casefold() != artist.casefold():
        return f"{title} — {artist}"
    return title or stream_name or artist or f"Sound {index:04d}"


def safe_stream_filename(stream: SoundStream) -> str:
    safe = re.sub(r"[^A-Za-z0-9._ ()\[\]{}-]+", "_", stream.name).strip(" ._")
    return safe or f"Sound_{stream.index:04d}"


def inspect_sound_bank(acb_path: Path, project_root: Path | None = None) -> SoundBank:
    acb_path = acb_path.resolve()
    cli = find_vgmstream(project_root)
    if cli is None:
        raise SoundToolError(
            "vgmstream-cli was not found. Place vgmstream-cli.exe in the tool's tools folder "
            "or add it to PATH."
        )
    awb_path = acb_path.with_suffix(".awb")
    if not awb_path.is_file():
        # Some banks use a differently cased or named AWB in the same directory.
        matches = [p for p in acb_path.parent.iterdir() if p.is_file() and p.suffix.casefold() == ".awb" and p.stem.casefold() == acb_path.stem.casefold()]
        awb_path = matches[0] if matches else None

    text = _metadata(cli, acb_path, 1)
    count = 1
    for pattern in _STREAM_COUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            count = max(1, int(match.group(1)))
            break

    first_stream_name = _first_match(_STREAM_NAME_PATTERNS, text)
    first_title = _first_match(_TITLE_PATTERNS, text)
    first_artist = _first_match(_ARTIST_PATTERNS, text)
    first_name = _display_name(first_stream_name, first_title, first_artist, 1)
    rate_match = _RATE_RE.search(text)
    channels_match = _CHANNELS_RE.search(text)
    samples_match = _SAMPLES_RE.search(text)
    first_rate = int(rate_match.group(1)) if rate_match else None
    first_channels = int(channels_match.group(1)) if channels_match else None
    first_samples = int(samples_match.group(1)) if samples_match else None

    streams = [
        SoundStream(
            index=i,
            name=(first_name if i == 1 else f"Sound {i:04d}"),
            sample_rate=(first_rate if i == 1 else None),
            channels=(first_channels if i == 1 else None),
            samples=(first_samples if i == 1 else None),
            title=(first_title if i == 1 else None),
            artist=(first_artist if i == 1 else None),
            metadata_loaded=(i == 1),
        )
        for i in range(1, count + 1)
    ]
    return SoundBank(acb_path, awb_path, streams, cli)


def read_stream_metadata(bank: SoundBank, stream: SoundStream) -> SoundStream:
    if stream.metadata_loaded:
        return stream
    text = _metadata(bank.vgmstream_path, bank.acb_path, stream.index)
    stream_name = _first_match(_STREAM_NAME_PATTERNS, text)
    title = _first_match(_TITLE_PATTERNS, text)
    artist = _first_match(_ARTIST_PATTERNS, text)
    rate_match = _RATE_RE.search(text)
    channels_match = _CHANNELS_RE.search(text)
    samples_match = _SAMPLES_RE.search(text)
    return SoundStream(
        stream.index,
        _display_name(stream_name, title, artist, stream.index),
        int(rate_match.group(1)) if rate_match else None,
        int(channels_match.group(1)) if channels_match else None,
        int(samples_match.group(1)) if samples_match else None,
        title,
        artist,
        True,
    )

def _resample_pcm16_wav(path: Path, target_rate: int = 44100) -> None:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        source_rate = reader.getframerate()
        frames = reader.getnframes()
        raw = reader.readframes(frames)
    if source_rate == target_rate:
        return
    if sample_width != 2:
        raise SoundToolError(f"Expected 16-bit PCM from vgmstream, got {sample_width * 8}-bit WAV")
    data = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels)
    source_length = data.shape[0]
    target_length = max(1, int(round(source_length * target_rate / source_rate)))
    source_positions = np.linspace(0.0, 1.0, source_length, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, target_length, endpoint=False)
    if channels == 1:
        resampled = np.interp(target_positions, source_positions, data.astype(np.float64))
    else:
        resampled = np.column_stack(
            [np.interp(target_positions, source_positions, data[:, channel].astype(np.float64)) for channel in range(channels)]
        )
    resampled = np.clip(np.rint(resampled), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(target_rate)
        writer.writeframes(resampled.tobytes())


def decode_stream(bank: SoundBank, stream_index: int, output_path: Path, target_rate: int = 44100) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        bank.vgmstream_path,
        ["-i", "-s", str(stream_index), "-o", str(output_path), str(bank.acb_path)],
        timeout=300,
    )
    if not output_path.is_file():
        raise SoundToolError(f"vgmstream did not create {output_path.name}")
    _resample_pcm16_wav(output_path, target_rate)
    return output_path


def decode_stream_to_temp(bank: SoundBank, stream_index: int, temp_dir: Path | None = None) -> Path:
    root = temp_dir or Path(tempfile.gettempdir()) / "UltimateXenoverseExtractor" / "audio_preview"
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{bank.acb_path.stem}_{stream_index:04d}.wav"
    return decode_stream(bank, stream_index, output, 44100)


def extract_all_streams(
    bank: SoundBank,
    output_dir: Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    total = len(bank.streams)
    for position, stream in enumerate(bank.streams, 1):
        if not stream.metadata_loaded:
            stream = read_stream_metadata(bank, stream)
            bank.streams[position - 1] = stream
        safe_name = safe_stream_filename(stream)
        output = output_dir / f"{stream.index:04d}_{safe_name}.wav"
        decode_stream(bank, stream.index, output, 44100)
        outputs.append(output)
        if progress:
            progress(position, total, output.name)
    return outputs
