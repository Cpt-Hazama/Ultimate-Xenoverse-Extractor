from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QTimer, QUrl
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QSpinBox,
    QTabBar,
    QToolButton,
    QWidget,
)


class UISoundManager(QObject):
    """Optional UI sounds with robust path discovery and deferred playback."""

    FILES = {
        "click": "00_click.wav",
        "cancel": "01_cancel.wav",
        "hover": "03_hover.wav",
        "page_change": "04_page_change.wav",
        "window_open": "05_window_open.wav",
        "window_close": "06_window_close.wav",
    }

    def __init__(self, sound_directory: Path, *, enabled: bool = True, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.enabled = bool(enabled)
        self._effects: dict[str, QSoundEffect] = {}
        self._pending: set[str] = set()
        self._last_hover_time = 0.0
        self._last_click_time = 0.0

        requested = Path(sound_directory).resolve()
        project_root = requested.parent
        package_dir = project_root / "ultimate_xenoverse_extractor"
        candidates = [
            requested,                         # project_root/sounds
            package_dir / "sounds",           # package/sounds
            Path.cwd() / "sounds",             # launch-directory/sounds
            Path(__file__).resolve().parent / "sounds",
            Path(__file__).resolve().parent.parent / "sounds",
        ]
        self._candidate_directories = list(dict.fromkeys(path.resolve() for path in candidates))
        self.active_directory: Path | None = next(
            (path for path in self._candidate_directories if any((path / f).is_file() for f in self.FILES.values())),
            None,
        )
        self._load_effects()

    @property
    def loaded_count(self) -> int:
        return len(self._effects)

    @property
    def searched_directories(self) -> tuple[Path, ...]:
        return tuple(self._candidate_directories)

    def _load_effects(self) -> None:
        if self.active_directory is None:
            return
        volumes = {
            "click": 0.30,
            "cancel": 0.32,
            "hover": 0.32,
            "page_change": 0.35,
            "window_open": 0.32,
            "window_close": 0.32,
        }
        for name, filename in self.FILES.items():
            path = self.active_directory / filename
            if not path.is_file():
                continue
            effect = QSoundEffect(self)
            effect.setLoopCount(1)
            effect.setVolume(volumes.get(name, 0.65))
            effect.statusChanged.connect(lambda n=name: self._on_status_changed(n))
            effect.setSource(QUrl.fromLocalFile(str(path)))
            self._effects[name] = effect

    def _on_status_changed(self, name: str) -> None:
        effect = self._effects.get(name)
        if effect is None:
            return
        if name in self._pending and effect.status() == QSoundEffect.Ready:
            self._pending.discard(name)
            QTimer.singleShot(0, effect.play)
        elif effect.status() == QSoundEffect.Error:
            self._pending.discard(name)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self._pending.clear()
            for effect in self._effects.values():
                effect.stop()

    def _play(self, name: str) -> None:
        if not self.enabled:
            return
        effect = self._effects.get(name)
        if effect is None:
            return
        status = effect.status()
        if status == QSoundEffect.Loading:
            self._pending.add(name)
            return
        if status != QSoundEffect.Ready:
            return
        if effect.isPlaying():
            effect.stop()
        effect.play()

    def play_click(self) -> None:
        now = time.monotonic()
        if now - self._last_click_time < 0.045:
            return
        self._last_click_time = now
        self._play("click")

    def play_cancel(self) -> None:
        self._play("cancel")

    def play_hover(self) -> None:
        now = time.monotonic()
        if now - self._last_hover_time < 0.10:
            return
        self._last_hover_time = now
        self._play("hover")

    def play_page_change(self) -> None:
        self._play("page_change")

    def play_window_open(self) -> None:
        self._play("window_open")

    def play_window_close(self) -> None:
        self._play("window_close")

    @staticmethod
    def _is_interactive(widget: object) -> bool:
        return isinstance(widget, QWidget) and widget.isEnabled() and isinstance(
            widget,
            (
                QAbstractButton,
                QAbstractItemView,
                QComboBox,
                QDoubleSpinBox,
                QLineEdit,
                QSlider,
                QSpinBox,
                QTabBar,
                QToolButton,
            ),
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        event_type = event.type()
        if event_type == QEvent.KeyPress:
            try:
                if event.key() == 0x01000000:  # Qt.Key_Escape
                    self.play_cancel()
            except AttributeError:
                pass
            return False

        if not self._is_interactive(watched):
            return False
        if event_type == QEvent.Enter:
            self.play_hover()
        elif event_type == QEvent.MouseButtonPress:
            self.play_click()
        return False
