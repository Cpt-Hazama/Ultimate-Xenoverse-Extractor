"""Bounded viewport-only per-frame pose cache.

EAN files are decoded once by app._load_cached_ean; this is a separate cache
for evaluated poses, so looping/scrubbing does not continually re-evaluate
hundreds of animation channels and inverse-bind matrices.

Only one active clip/composition is cached. Changing character, body animation,
or facial/lip overlays discards the old allocation immediately. Exporters keep
sampling source curves directly at full precision.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

MAX_CACHE_BYTES = 96 * 1024 * 1024


@dataclass(slots=True)
class DensePoseCache:
    frames: int
    bones: int
    pose: np.ndarray  # (frame, bone, [position3, quaternion4, scale3])
    skin: np.ndarray  # (frame, bone, 4, 4)
    valid: np.ndarray
    filled: int = 0

    @classmethod
    def create(cls, frames: int, bones: int, max_bytes: int = MAX_CACHE_BYTES):
        frames, bones = int(frames), int(bones)
        # Include the bitmap and disallow zero-geometry or unreasonably long
        # animations rather than accidentally reserving gigabytes.
        nbytes = frames * bones * (10 + 16) * 4 + frames
        if frames < 1 or bones < 1 or nbytes > max_bytes:
            return None
        return cls(
            frames, bones,
            np.empty((frames, bones, 10), dtype=np.float32),
            np.empty((frames, bones, 4, 4), dtype=np.float32),
            np.zeros(frames, dtype=np.bool_),
        )

    @property
    def allocated_bytes(self) -> int:
        return self.pose.nbytes + self.skin.nbytes + self.valid.nbytes

    def get(self, frame: float):
        index = int(frame)
        if frame != index or index < 0 or index >= self.frames or not self.valid[index]:
            return None
        return self.pose[index], self.skin[index]

    def store(self, frame: float, positions, rotations, scales, matrices) -> None:
        index = int(frame)
        if frame != index or index < 0 or index >= self.frames or self.valid[index]:
            return
        if len(positions) != self.bones or len(matrices) != self.bones:
            return
        dest = self.pose[index]
        dest[:, :3] = positions
        dest[:, 3:7] = rotations
        dest[:, 7:10] = scales
        self.skin[index] = matrices
        self.valid[index] = True
        self.filled += 1
