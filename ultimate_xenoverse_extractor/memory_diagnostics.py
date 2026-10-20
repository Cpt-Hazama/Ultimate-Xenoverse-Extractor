"""Opt-in process-memory logger for Python/Qt/OpenGL playback leaks.

GPU and driver allocations are not visible to tracemalloc; the difference
between Windows private bytes and the traced Python heap is diagnostic.
No third-party dependency is needed on Windows.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
import tracemalloc
from pathlib import Path


def _windows_process_memory() -> tuple[int, int]:
    from ctypes import wintypes
    size_t = ctypes.c_size_t

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", size_t),
            ("WorkingSetSize", size_t),
            ("QuotaPeakPagedPoolUsage", size_t),
            ("QuotaPagedPoolUsage", size_t),
            ("QuotaPeakNonPagedPoolUsage", size_t),
            ("QuotaNonPagedPoolUsage", size_t),
            ("PagefileUsage", size_t),
            ("PeakPagefileUsage", size_t),
            ("PrivateUsage", size_t),
        ]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD)
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
    return int(pmc.WorkingSetSize), int(pmc.PrivateUsage)


def process_memory() -> tuple[int, int]:
    if sys.platform == 'win32':
        return _windows_process_memory()
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        return int(info.rss), int(getattr(info, 'vms', info.rss))
    except ImportError:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return rss, rss


class MemoryLogger:
    def __init__(self, path: Path):
        self.path = path
        self.base_private: int | None = None
        self.base_python: int | None = None
        self.started = time.monotonic()

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tracemalloc.start(5)
        self.started = time.monotonic()
        self.base_private = None
        self.base_python = None
        with self.path.open('w', encoding='utf-8') as output:
            output.write('time_s\trss_MB\tprivate_MB\tprivate_delta_MB\tpython_heap_MB\tpython_delta_MB\tpose_cache_MB\tcached_frames\ttotal_frames\tframe\tsurfaces\tGL_objects\tanimation\n')

    def stop(self):
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def sample(self, viewport) -> str:
        rss, private = process_memory()
        python_bytes = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        if self.base_private is None:
            self.base_private = private
            self.base_python = python_bytes
        cache = viewport._pose_cache
        surfaces = [surface for part in viewport.parts.values() for surface in part.surfaces]
        gl_objects = sum(sum(obj is not None for obj in (
            s.gpu.vao, s.gpu.position_vbo, s.gpu.static_vbo,
            s.gpu.ibo, s.gpu.texture_obj, s.gpu.lightwarp_obj,
            s.gpu.palette_ubo,
        )) for s in surfaces)
        to_mb = lambda value: value / 1048576
        info = (
            f'{time.monotonic()-self.started:.1f}\t{to_mb(rss):.1f}\t'
            f'{to_mb(private):.1f}\t{to_mb(private-self.base_private):+.1f}\t'
            f'{to_mb(python_bytes):.1f}\t{to_mb(python_bytes-self.base_python):+.1f}\t'
            f'{to_mb(cache.allocated_bytes) if cache else 0:.1f}\t'
            f'{cache.filled if cache else 0}\t{cache.frames if cache else 0}\t'
            f'{viewport.animation_frame:g}\t{len(surfaces)}\t{gl_objects}\t'
            f'{viewport.current_animation.name if viewport.current_animation else "(none)"}\n'
        )
        with self.path.open('a', encoding='utf-8') as output:
            output.write(info)
        return (f'RAM {to_mb(private):.0f} MB '
                f'({to_mb(private-self.base_private):+.0f}) | '
                f'Python {to_mb(python_bytes):.0f} MB '
                f'({to_mb(python_bytes-self.base_python):+.0f})')
