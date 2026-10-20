from __future__ import annotations

import sys
import gc
from pathlib import Path
import re
from collections import defaultdict

try:
    from PySide6.QtCore import Qt, QSize, QTimer, QUrl
    from PySide6.QtGui import QAction, QColor, QFont, QBrush, QPixmap, QIcon
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
        QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
        QProgressBar, QPushButton, QSplitter, QStatusBar, QTabWidget, QTableWidget,
        QTableWidgetItem, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QSizePolicy, QSlider, QDoubleSpinBox, QComboBox,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PySide6 is required. Run: pip install -r requirements.txt") from exc

from .id_lists import load_all_id_lists
from .models import AssetKind, LibraryEntry
from .scanner import build_library, summarize_assets
from .parsers import EmbFormatError, EmdFormatError, EmmFormatError, EskFormatError, EanFormatError, load_emb, load_emd, load_emm, load_esk, load_ean
from .viewport import ModelViewport
from .memory_diagnostics import MemoryLogger
from .settings import load_settings, save_settings
from .source_export import ExportOptions, export_character
from .ui_sounds import UISoundManager
from .sound_tools import (
    SoundBank, SoundStream, SoundToolError, decode_stream, decode_stream_to_temp,
    extract_all_streams, inspect_sound_bank, read_stream_metadata, safe_stream_filename,
)
from .cac_profiles import (
    build_cac_entries, load_profiles, parse_save_xml, resolve_skill_entries, save_profiles,
)

APP_NAME = "Ultimate Xenoverse Extractor"
BLUE = QColor("#78bfff")
RED = QColor("#f07b7b")
NEUTRAL = QColor("#d5d9df")


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.id_dir = project_root / "_IDLists"
        self.id_lists = load_all_id_lists(self.id_dir)
        self.cac_store_path = project_root / "userdata" / "cac_profiles.json"
        self.cac_profiles = load_profiles(self.cac_store_path)
        self.entries: list[LibraryEntry] = []
        self._character_skill_index: dict[str, list[LibraryEntry]] = {}
        self._ean_cache: dict[Path, object] = {}
        self.current_entry: LibraryEntry | None = None
        self.current_character_entry: LibraryEntry | None = None
        self.additional_animation_entries: list[LibraryEntry] = []
        self.additional_animation_paths: set[Path] = set()
        self.additional_camera_sources: list[tuple[object, object, str, Path]] = []
        self.animation_sources: list[tuple[object, object, str, Path, bool]] = []
        self.settings = load_settings()
        self.portrait_cache: dict[str, QIcon] = {}
        self._portrait_queue: list[tuple[QTreeWidgetItem, Path, int]] = []
        self._portrait_generation = 0
        self._portrait_batch_timer = QTimer(self)
        self._portrait_batch_timer.setSingleShot(True)
        self._portrait_batch_timer.timeout.connect(self._process_portrait_batch)
        self.current_sound_entry: LibraryEntry | None = None
        self.current_sound_bank: SoundBank | None = None
        self.sound_audio_output = QAudioOutput(self)
        self.sound_audio_output.setVolume(0.8)
        self.sound_player = QMediaPlayer(self)
        self.sound_player.setAudioOutput(self.sound_audio_output)
        self._sound_metadata_generation = 0
        self._sound_metadata_index = 0
        self._sound_metadata_timer = QTimer(self)
        self._sound_metadata_timer.setSingleShot(True)
        self._sound_metadata_timer.timeout.connect(self._load_next_sound_metadata)
        self.ui_sounds = UISoundManager(
            project_root / "sounds",
            enabled=bool(self.settings.get("ui_sounds_enabled", True)),
            parent=self,
        )
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self.ui_sounds)

        self.setWindowTitle(f"{APP_NAME} - 0.2.0")
        self.resize(1500, 900)
        self._make_toolbar()
        self._make_ui()
        self._apply_style()
        self.setStatusBar(QStatusBar())
        self._memory_logger = MemoryLogger(project_root / "userdata" / "memory_diagnostics.tsv")
        self._memory_timer = QTimer(self)
        self._memory_timer.setInterval(2000)
        self._memory_timer.timeout.connect(self._sample_memory)
        self._memory_label = QLabel("")
        self.statusBar().addPermanentWidget(self._memory_label)

        previous = self.settings.get("asset_directory")
        if previous and Path(previous).is_dir():
            self.load_directory(Path(previous))
        else:
            self.choose_directory(first_run=True)

    def _make_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        open_action = QAction("Open Asset Directory", self)
        open_action.triggered.connect(self.choose_directory)
        toolbar.addAction(open_action)

        refresh_action = QAction("Refresh Library", self)
        refresh_action.triggered.connect(self.refresh)
        toolbar.addAction(refresh_action)

        # import_cac_action = QAction("Import XV2 Save XML", self)
        # import_cac_action.triggered.connect(self.import_cac_save_xml)
        # toolbar.addAction(import_cac_action)

        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        toolbar.addAction(reset_layout_action)

        self.ui_sounds_action = QAction("UI Sounds", self)
        self.ui_sounds_action.setCheckable(True)
        self.ui_sounds_action.setChecked(self.ui_sounds.enabled)
        self.ui_sounds_action.toggled.connect(self.set_ui_sounds_enabled)
        toolbar.addAction(self.ui_sounds_action)
        toolbar.addSeparator()

        self.memory_action = QAction("Memory Diagnostics", self)
        self.memory_action.setCheckable(True)
        self.memory_action.setToolTip("Track Windows private RAM versus Python allocations while animations play; writes userdata/memory_diagnostics.tsv")
        self.memory_action.toggled.connect(self._toggle_memory_diagnostics)
        toolbar.addAction(self.memory_action)

        diagnostics_menu = self.menuBar().addMenu("Diagnostics")
        diagnostics_menu.addAction(self.memory_action)
        self.freeze_gpu_action = QAction("Freeze bone / mesh uploads (isolate GPU memory)", self)
        self.freeze_gpu_action.setCheckable(True)
        self.freeze_gpu_action.toggled.connect(lambda enabled: setattr(self.viewport, "freeze_gpu_uploads", enabled))
        diagnostics_menu.addAction(self.freeze_gpu_action)
        self.skip_models_action = QAction("Skip mesh drawing (isolate render driver)", self)
        self.skip_models_action.setCheckable(True)
        self.skip_models_action.toggled.connect(lambda enabled: setattr(self.viewport, "skip_model_draw", enabled))
        diagnostics_menu.addAction(self.skip_models_action)

        self.path_label = QLabel(" No asset directory selected ")
        self.path_label.setObjectName("pathLabel")
        toolbar.addWidget(self.path_label)

    def _toggle_memory_diagnostics(self, enabled: bool) -> None:
        if enabled:
            self._memory_logger.start()
            self._sample_memory()
            self._memory_timer.start()
            self.statusBar().showMessage(
                "Logging memory to userdata/memory_diagnostics.tsv", 6000
            )
        else:
            self._memory_timer.stop()
            self._memory_logger.stop()
            self._memory_label.setText("")

    def _sample_memory(self) -> None:
        try:
            self._memory_label.setText(self._memory_logger.sample(self.viewport))
        except Exception as exc:
            self._memory_label.setText(f"Memory diagnostics error: {exc}")
            self._memory_timer.stop()

    def _make_ui(self) -> None:
        self.root_split = QSplitter(Qt.Horizontal)
        self.root_split.setChildrenCollapsible(False)
        self.root_split.setHandleWidth(10)
        self.root_split.setOpaqueResize(True)
        root_split = self.root_split
        self.setCentralWidget(root_split)

        # Left: library
        left = QWidget()
        self.library_panel = left
        left.setMinimumWidth(260)
        left.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left)
        heading = QLabel("ASSET LIBRARY")
        heading.setObjectName("sectionTitle")
        left_layout.addWidget(heading)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search names, IDs, or folders…")
        self.search.textChanged.connect(self.apply_filter)
        left_layout.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(48, 48))
        self.tree.setIndentation(8)
        self.tree.itemSelectionChanged.connect(self.library_selection_changed)
        left_layout.addWidget(self.tree)
        legend = QLabel("Blue: playable / CaC usable    Red: non-usable    Gray: unknown")
        legend.setObjectName("muted")
        legend.setWordWrap(True)
        left_layout.addWidget(legend)
        root_split.addWidget(left)

        # Center: viewport and details
        self.center_split = QSplitter(Qt.Vertical)
        self.center_split.setChildrenCollapsible(False)
        self.center_split.setHandleWidth(9)
        self.center_split.setOpaqueResize(True)
        self.center_split.setMinimumWidth(480)
        self.center_split.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        center = self.center_split
        viewport_panel = QWidget()
        viewport_layout = QVBoxLayout(viewport_panel)
        viewport_layout.setContentsMargins(8, 8, 8, 8)
        viewport_bar = QHBoxLayout()
        viewport_title = QLabel("3D VIEWPORT")
        viewport_title.setObjectName("sectionTitle")
        viewport_title.setToolTip("Ctrl + left mouse drag in the viewport to reposition the cel-shading light.")
        viewport_bar.addWidget(viewport_title)
        viewport_bar.addStretch()
        self.solid_box = QCheckBox("Solid")
        self.solid_box.setChecked(True)
        self.wire_box = QCheckBox("Wireframe")
        self.grid_box = QCheckBox("Grid")
        self.grid_box.setChecked(True)
        self.bones_box = QCheckBox("Skeleton")
        self.bones_box.setChecked(True)
        for box in (self.solid_box, self.wire_box, self.grid_box, self.bones_box):
            viewport_bar.addWidget(box)
        fit_button = QPushButton("Fit")
        fit_button.setMaximumWidth(70)
        reset_button = QPushButton("Reset")
        reset_button.setMaximumWidth(70)
        clear_button = QPushButton("Clear")
        clear_button.setMaximumWidth(70)
        viewport_bar.addWidget(fit_button)
        viewport_bar.addWidget(reset_button)
        viewport_bar.addWidget(clear_button)
        viewport_layout.addLayout(viewport_bar)
        self.viewport = ModelViewport()
        self.viewport.statusChanged.connect(lambda text: self.statusBar().showMessage(text, 5000))
        viewport_layout.addWidget(self.viewport)
        self.solid_box.toggled.connect(lambda value: self._set_viewport_option("show_solid", value))
        self.wire_box.toggled.connect(lambda value: self._set_viewport_option("show_wireframe", value))
        self.grid_box.toggled.connect(lambda value: self._set_viewport_option("show_grid", value))
        self.bones_box.toggled.connect(lambda value: self._set_viewport_option("show_skeleton", value))
        fit_button.clicked.connect(self.viewport.fit_scene)
        reset_button.clicked.connect(self.viewport.reset_camera)
        clear_button.clicked.connect(self.clear_viewport)
        center.addWidget(viewport_panel)
        info = QWidget()
        info_layout = QVBoxLayout(info)
        self.character_title = QLabel("No asset selected")
        self.character_title.setObjectName("characterTitle")
        self.character_meta = QLabel("")
        self.character_meta.setObjectName("muted")
        info_layout.addWidget(self.character_title)
        info_layout.addWidget(self.character_meta)
        self.assignment_table = QTableWidget(0, 3)
        self.assignment_table.setHorizontalHeaderLabels(["Model", "Assigned Skeleton", "Role"])
        self.assignment_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.assignment_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.assignment_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.assignment_table.setEditTriggers(QTableWidget.NoEditTriggers)
        info_layout.addWidget(self.assignment_table)
        center.addWidget(info)
        center.setSizes([600, 250])
        root_split.addWidget(center)

        # Right: files / animations / export
        right = QWidget()
        self.inspector_panel = right
        right.setMinimumWidth(340)
        right.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right)
        right_heading = QLabel("ASSET INSPECTOR")
        right_heading.setObjectName("sectionTitle")
        right_layout.addWidget(right_heading)
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        # A non-expanding, non-eliding bar preserves full labels and uses Qt's
        # native left/right scroll buttons instead of compressing all tabs.
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.asset_list = QListWidget()
        self.asset_list.setAlternatingRowColors(True)
        self.asset_list.itemSelectionChanged.connect(self.asset_selected)
        self.tabs.addTab(self.asset_list, "Files")

        models_tab = QWidget()
        models_layout = QVBoxLayout(models_tab)
        model_buttons = QHBoxLayout()
        load_all_button = QPushButton("Load All")
        unload_all_button = QPushButton("Unload All")
        model_buttons.addWidget(load_all_button)
        model_buttons.addWidget(unload_all_button)
        models_layout.addLayout(model_buttons)
        self.model_list = QListWidget()
        self.model_list.itemChanged.connect(self.model_visibility_changed)
        self.model_list.itemDoubleClicked.connect(self.focus_model_item)
        models_layout.addWidget(self.model_list)
        self.model_meta = QLabel("Check models to load them into the native viewport.")
        self.model_meta.setObjectName("muted")
        self.model_meta.setWordWrap(True)
        models_layout.addWidget(self.model_meta)
        load_all_button.clicked.connect(lambda: self.set_all_models_checked(True))
        unload_all_button.clicked.connect(lambda: self.set_all_models_checked(False))
        self.tabs.addTab(models_tab, "Models")

        material_tab = QWidget()
        material_layout = QVBoxLayout(material_tab)
        self.material_meta = QLabel("Select an EMM file to inspect its shaders, colors, and parameters.")
        self.material_meta.setObjectName("muted")
        self.material_meta.setWordWrap(True)
        material_layout.addWidget(self.material_meta)
        self.material_split = QSplitter(Qt.Vertical)
        self.material_split.setChildrenCollapsible(False)
        self.material_split.setHandleWidth(6)
        self.material_tree = QTreeWidget()
        self.material_tree.setHeaderLabels(["Material", "Shader", "Parameters"])
        self.material_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.material_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.material_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.material_tree.itemSelectionChanged.connect(self.material_selection_changed)
        self.material_split.addWidget(self.material_tree)
        parameter_panel = QWidget()
        parameter_layout = QVBoxLayout(parameter_panel)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.material_swatch = QLabel("Material preview")
        self.material_swatch.setAlignment(Qt.AlignCenter)
        self.material_swatch.setMinimumHeight(42)
        self.material_swatch.setObjectName("materialSwatch")
        parameter_layout.addWidget(self.material_swatch)
        self.parameter_table = QTableWidget(0, 4)
        self.parameter_table.setHorizontalHeaderLabels(["Parameter", "Type", "Level", "Value"])
        self.parameter_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.parameter_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.parameter_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.parameter_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.parameter_table.setEditTriggers(QTableWidget.NoEditTriggers)
        parameter_layout.addWidget(self.parameter_table)
        self.material_split.addWidget(parameter_panel)
        self.material_split.setSizes([300, 340])
        material_layout.addWidget(self.material_split)
        self.material_tab = material_tab
        self.current_emm = None
        self.tabs.addTab(material_tab, "Materials")

        texture_tab = QWidget()
        texture_layout = QVBoxLayout(texture_tab)
        self.texture_meta = QLabel("Select an EMB file to inspect its contained DDS textures.")
        self.texture_meta.setObjectName("muted")
        self.texture_meta.setWordWrap(True)
        texture_layout.addWidget(self.texture_meta)
        self.texture_list = QListWidget()
        self.texture_list.currentRowChanged.connect(self.texture_selection_changed)
        texture_layout.addWidget(self.texture_list, 1)
        self.texture_preview = QLabel("No texture selected")
        self.texture_preview.setAlignment(Qt.AlignCenter)
        self.texture_preview.setMinimumHeight(180)
        self.texture_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        texture_layout.addWidget(self.texture_preview, 2)
        self.texture_tab = texture_tab
        self.current_emb = None
        self.tabs.addTab(texture_tab, "Textures")

        skeleton_tab = QWidget()
        skeleton_layout = QVBoxLayout(skeleton_tab)
        self.skeleton_meta = QLabel("Select an ESK file to inspect its native hierarchy.")
        self.skeleton_meta.setObjectName("muted")
        self.skeleton_meta.setWordWrap(True)
        skeleton_layout.addWidget(self.skeleton_meta)
        self.skeleton_tree = QTreeWidget()
        self.skeleton_tree.setHeaderLabels(["Bone", "Index", "Parent", "IK"])
        self.skeleton_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.skeleton_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.skeleton_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.skeleton_tree.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        skeleton_layout.addWidget(self.skeleton_tree)
        self.skeleton_tab = skeleton_tab
        self.tabs.addTab(skeleton_tab, "Skeleton")
        animation_tab = QWidget()
        animation_layout = QVBoxLayout(animation_tab)
        self.animation_meta = QLabel("Select an EAN file or choose an animation below.")
        self.animation_meta.setObjectName("muted")
        self.animation_meta.setWordWrap(True)
        animation_layout.addWidget(self.animation_meta)
        self.animation_list = QListWidget()
        self.animation_list.currentRowChanged.connect(self.animation_selection_changed)
        animation_layout.addWidget(self.animation_list, 1)
        face_row = QHBoxLayout()
        face_row.addWidget(QLabel("Face animation"))
        self.face_combo = QComboBox()
        self.face_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.face_combo.setMinimumContentsLength(18)
        self.face_combo.addItem("None", -1)
        face_row.addWidget(self.face_combo, 1)
        animation_layout.addLayout(face_row)

        lip_row = QHBoxLayout()
        lip_row.addWidget(QLabel("Lip animation"))
        self.lip_combo = QComboBox()
        self.lip_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.lip_combo.setMinimumContentsLength(18)
        self.lip_combo.addItem("None", -1)
        lip_row.addWidget(self.lip_combo, 1)
        animation_layout.addLayout(lip_row)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera animation"))
        self.camera_combo = QComboBox()
        self.camera_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.setMinimumContentsLength(18)
        self.camera_combo.addItem("None", -1)
        camera_row.addWidget(self.camera_combo, 1)
        animation_layout.addLayout(camera_row)

        camera_options_row = QHBoxLayout()
        self.camera_box = QCheckBox("Follow camera")
        self.camera_box.setChecked(False)
        camera_options_row.addWidget(self.camera_box)
        camera_options_row.addStretch(1)
        animation_layout.addLayout(camera_options_row)

        self.overlay_meta = QLabel("Facial: none • Lip: none • Camera: none")
        self.overlay_meta.setObjectName("muted")
        self.overlay_meta.setWordWrap(True)
        animation_layout.addWidget(self.overlay_meta)
        playback_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.disable_animation_button = QPushButton("T-Pose")
        self.disable_animation_button.setToolTip("Disable animation playback and return the model to its bind/T-pose.")
        self.remove_additional_button = QPushButton("Remove Additional Animations")
        self.remove_additional_button.setToolTip("Remove every skill/additional EAN appended to the current character.")
        self.loop_box = QCheckBox("Loop")
        self.loop_box.setChecked(True)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix("×")
        playback_row.addWidget(self.play_button)
        playback_row.addWidget(self.stop_button)
        playback_row.addWidget(self.disable_animation_button)
        playback_row.addWidget(self.remove_additional_button)
        playback_row.addWidget(self.loop_box)
        playback_row.addWidget(QLabel("Speed"))
        playback_row.addWidget(self.speed_spin)
        animation_layout.addLayout(playback_row)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.animation_frame_changed)
        animation_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel("Frame 0 / 0")
        self.frame_label.setObjectName("muted")
        animation_layout.addWidget(self.frame_label)
        self.animation_tab = animation_tab
        self.current_ean = None
        self.face_ean = None
        self.lip_ean = None
        self.camera_ean = None
        self.current_animation = None
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(16)
        self.animation_timer.timeout.connect(self.advance_animation)
        self.play_button.clicked.connect(self.toggle_animation_playback)
        self.stop_button.clicked.connect(self.stop_animation)
        self.disable_animation_button.clicked.connect(self.disable_animation)
        self.remove_additional_button.clicked.connect(self.remove_additional_animations)
        self.face_combo.currentIndexChanged.connect(self.refresh_animation_overlays)
        self.lip_combo.currentIndexChanged.connect(self.refresh_animation_overlays)
        self.camera_box.toggled.connect(self.refresh_animation_overlays)
        self.camera_combo.currentIndexChanged.connect(self.refresh_animation_overlays)
        self.tabs.addTab(animation_tab, "Animations")

        sound_tab = QWidget()
        sound_layout = QVBoxLayout(sound_tab)
        self.sound_meta = QLabel("Select an ACB sound bank from the library.")
        self.sound_meta.setObjectName("muted")
        self.sound_meta.setWordWrap(True)
        sound_layout.addWidget(self.sound_meta)
        self.sound_list = QListWidget()
        self.sound_list.currentRowChanged.connect(self.sound_stream_selected)
        self.sound_list.itemDoubleClicked.connect(lambda _item: self.play_selected_sound())
        sound_layout.addWidget(self.sound_list, 1)
        sound_buttons = QHBoxLayout()
        self.sound_play_button = QPushButton("Play")
        self.sound_stop_button = QPushButton("Stop")
        self.sound_extract_button = QPushButton("Extract Selected")
        self.sound_extract_all_button = QPushButton("Extract All")
        self.sound_play_button.clicked.connect(self.play_selected_sound)
        self.sound_stop_button.clicked.connect(self.stop_sound)
        self.sound_extract_button.clicked.connect(self.extract_selected_sound)
        self.sound_extract_all_button.clicked.connect(self.extract_all_sounds)
        for button in (self.sound_play_button, self.sound_stop_button, self.sound_extract_button, self.sound_extract_all_button):
            button.setEnabled(False)
            sound_buttons.addWidget(button)
        sound_layout.addLayout(sound_buttons)
        self.sound_tab = sound_tab
        self.tabs.addTab(sound_tab, "Sounds")

        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        export_layout.addWidget(QLabel("Source extraction pipeline"))
        self.meshes_box = QCheckBox("Export separate SMD mesh per model")
        self.meshes_box.setChecked(True)
        self.anim_box = QCheckBox("Export SMD animations")
        self.anim_box.setChecked(True)
        self.texture_box = QCheckBox("Convert textures to TGA")
        self.texture_box.setChecked(True)
        self.event_box = QCheckBox("Write animation events JSON")
        self.event_box.setChecked(True)
        for box in (self.meshes_box, self.anim_box, self.texture_box, self.event_box):
            export_layout.addWidget(box)
        export_layout.addStretch()
        self.extract_button = QPushButton("Extract for Source")
        self.extract_button.setEnabled(False)
        self.extract_button.clicked.connect(self.extract_character)
        export_layout.addWidget(self.extract_button)
        self.tabs.addTab(export_tab, "Extract")
        self.tabs.currentChanged.connect(lambda _index: self.ui_sounds.play_page_change())
        right_layout.addWidget(self.tabs)
        root_split.addWidget(right)
        # All three regions remain user-resizable. The center takes most extra
        # space, but it is no longer allowed to crush the side panels when a
        # character populates large tables/lists.
        root_split.setStretchFactor(0, 1)
        root_split.setStretchFactor(1, 3)
        root_split.setStretchFactor(2, 1)
        root_split.setCollapsible(0, False)
        root_split.setCollapsible(1, False)
        root_split.setCollapsible(2, False)
        self.restore_layout()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #17191d; color: #e8eaed; font-family: 'Segoe UI'; font-size: 10pt; }
            QToolBar { background: #202329; border-bottom: 1px solid #343942; spacing: 8px; padding: 6px; }
            QLineEdit, QTreeWidget, QListWidget, QTableWidget, QTabWidget::pane {
                background: #111317; border: 1px solid #343942; border-radius: 5px;
            }
            QLineEdit { padding: 8px; }
            QTreeWidget::item, QListWidget::item { padding: 5px; }
            QTreeWidget::item:selected, QListWidget::item:selected { background: #31577a; }
            QHeaderView::section { background: #242830; color: #e8eaed; padding: 7px; border: 0; border-right: 1px solid #343942; }
            QTabBar::tab { background: #242830; min-width: 84px; padding: 8px 13px; margin-right: 2px; }
            QTabBar::scroller { width: 26px; }
            QTabBar QToolButton { background: #287ac0; border: 1px solid #343942; border-radius: 3px; }
            QTabBar QToolButton:disabled { background: #25282e; }
            QTabBar::tab:selected { background: #31577a; }
            QPushButton { background: #287ac0; border: 0; border-radius: 5px; padding: 10px; font-weight: 600; }
            QPushButton:disabled { background: #3b4048; color: #8d939c; }
            QFrame { background: #101216; border: 1px solid #343942; border-radius: 6px; }
            QLabel#sectionTitle { font-size: 11pt; font-weight: 700; letter-spacing: 1px; }
            QLabel#characterTitle { font-size: 16pt; font-weight: 700; }
            QLabel#muted { color: #9ca3ad; }
            QLabel#pathLabel { color: #b9c0c9; }
            QLabel#materialSwatch { border: 1px solid #4a515c; border-radius: 5px; font-weight: 600; }
            QSplitter::handle { background: #303640; margin: 1px; }
            QSplitter::handle:hover, QSplitter::handle:pressed { background: #4a8bc2; }
            QSplitter::handle:horizontal { width: 10px; }
            QSplitter::handle:vertical { height: 9px; }
        """)

    def set_ui_sounds_enabled(self, enabled: bool) -> None:
        self.ui_sounds.set_enabled(enabled)
        self.settings["ui_sounds_enabled"] = bool(enabled)
        save_settings(self.settings)
        if enabled:
            self.ui_sounds.play_click()
            loaded = self.ui_sounds.loaded_count
            location = self.ui_sounds.active_directory
            if loaded:
                self.statusBar().showMessage(
                    f"UI sounds enabled", 6000
                )
            else:
                self.statusBar().showMessage(
                    "UI sounds enabled, but no WAV files were found or decoded. Check the sounds folder.",
                    8000,
                )
        else:
            self.statusBar().showMessage("UI sounds disabled", 3000)

    def restore_layout(self) -> None:
        root_sizes = self.settings.get("root_split_sizes")
        center_sizes = self.settings.get("center_split_sizes")
        if isinstance(root_sizes, list) and len(root_sizes) == 3:
            self.root_split.setSizes([int(value) for value in root_sizes])
        else:
            self.root_split.setSizes([300, 860, 400])
        if isinstance(center_sizes, list) and len(center_sizes) == 2:
            self.center_split.setSizes([int(value) for value in center_sizes])
        else:
            self.center_split.setSizes([620, 250])
        material_sizes = self.settings.get("material_split_sizes")
        if isinstance(material_sizes, list) and len(material_sizes) == 2:
            self.material_split.setSizes([int(value) for value in material_sizes])

    def reset_layout(self) -> None:
        self.root_split.setSizes([300, 860, 400])
        self.center_split.setSizes([620, 250])
        self.material_split.setSizes([300, 340])
        self.statusBar().showMessage("Layout reset", 3000)

    def closeEvent(self, event) -> None:
        self._memory_timer.stop()
        self._memory_logger.stop()
        self.stop_sound()
        self.ui_sounds.play_window_close()
        self.settings["root_split_sizes"] = self.root_split.sizes()
        self.settings["center_split_sizes"] = self.center_split.sizes()
        self.settings["material_split_sizes"] = self.material_split.sizes()
        save_settings(self.settings)
        super().closeEvent(event)

    def choose_directory(self, checked=False, first_run: bool = False) -> None:
        title = "Select your extracted Xenoverse 2 data folder" if first_run else "Open extracted Xenoverse 2 assets"
        start = self.settings.get("asset_directory", str(self.project_root / "_ExampleCharacterFolder"))
        selected = QFileDialog.getExistingDirectory(self, title, start)
        if selected:
            self.load_directory(Path(selected))
        elif first_run:
            example = self.project_root / "_ExampleCharacterFolder"
            if example.is_dir():
                self.load_directory(example)

    def load_directory(self, directory: Path) -> None:
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.entries = build_library(directory, self.id_lists)
            self.entries.extend(build_cac_entries(directory, self.cac_profiles, self.id_lists))
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not scan the directory:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._ean_cache.clear()
        skill_index: dict[str, list[LibraryEntry]] = defaultdict(list)
        actual_character_ids = {e.code.upper() for e in self.entries if e.category == "Characters"}
        for candidate in self.entries:
            if not candidate.category.startswith("Skills"):
                continue
            if not any(asset.kind in (AssetKind.ANIMATION, AssetKind.CAMERA) for asset in candidate.assets):
                continue
            # Skill folders are named 034_GOD_GODS, for example; the middle
            # token is the character ID. Also recognize EAN suffixes like
            # 1080_CMN_POW_GOK.ean which are character-specific overrides.
            folder_ids = set(candidate.code.upper().split("_"))
            file_ids = set()
            for asset in candidate.assets:
                if asset.kind in (AssetKind.ANIMATION, AssetKind.CAMERA):
                    file_ids.update(asset.path.name.upper().replace(".CAM.EAN", "").replace(".EAN", "").split("_"))
            for code in folder_ids | file_ids:
                if code in actual_character_ids:
                    skill_index[code].append(candidate)
        self._character_skill_index = dict(skill_index)
        self.settings["asset_directory"] = str(directory)
        save_settings(self.settings)
        self.path_label.setText(f" {directory} ")
        self.populate_tree()
        cac_count = sum(1 for entry in self.entries if entry.category == "CaC")
        suffix = f" including {cac_count} imported CaC(s)" if cac_count else ""
        self.statusBar().showMessage(f"Indexed {len(self.entries)} folders{suffix}", 5000)

    def import_cac_save_xml(self) -> None:
        start = self.settings.get("cac_xml_directory") or self.settings.get("asset_directory") or str(self.project_root)
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import Xenoverse 2 save XML",
            str(start),
            "Xenoverse Save XML (*.xml);;XML files (*.xml);;All files (*.*)",
        )
        if not selected:
            return
        path = Path(selected)
        try:
            profiles = parse_save_xml(path)
        except (OSError, ValueError, Exception) as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not import the save XML:\n{exc}")
            return
        if not profiles:
            QMessageBox.information(self, APP_NAME, "No CaC characters were found in that XML.")
            return
        self.cac_profiles = profiles
        save_profiles(self.cac_store_path, profiles)
        self.settings["cac_xml_directory"] = str(path.parent)
        save_settings(self.settings)
        current = self.settings.get("asset_directory")
        if current and Path(current).is_dir():
            self.load_directory(Path(current))
        QMessageBox.information(
            self,
            APP_NAME,
            f"Imported {len(profiles)} CaC character(s).\n\n"
            f"They are stored in:\n{self.cac_store_path}",
        )

    def refresh(self) -> None:
        current = self.settings.get("asset_directory")
        if current and Path(current).is_dir():
            self.id_lists = load_all_id_lists(self.id_dir)
            self.cac_profiles = load_profiles(self.cac_store_path)
            self.load_directory(Path(current))

    def populate_tree(self) -> None:
        # Building the tree must remain cheap. Portrait EMBs are decoded later in
        # tiny event-loop batches so hundreds of portraits cannot block startup.
        self._portrait_generation += 1
        generation = self._portrait_generation
        self._portrait_batch_timer.stop()
        self._portrait_queue.clear()
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        # Sort on the resolved display name, never the internal three-letter ID.
        # Preserve original indices so Qt.UserRole still locates self.entries.
        ordered_entries = sorted(enumerate(self.entries), key=lambda pair: (
            pair[1].category.casefold(),
            re.sub(r"\s*\[[^]]+\]\s*$", "", pair[1].label).casefold(),
            pair[1].code.casefold(),
        ))
        for index, entry in ordered_entries:
            parent = groups.get(entry.category)
            if parent is None:
                parent = QTreeWidgetItem([entry.category])
                parent.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
                self.tree.addTopLevelItem(parent)
                groups[entry.category] = parent
            item = QTreeWidgetItem([entry.label])
            item.setData(0, Qt.UserRole, index)
            item.setToolTip(0, str(entry.path))
            item.setForeground(0, QBrush(BLUE if entry.usable is True else RED if entry.usable is False else NEUTRAL))
            portrait_path = entry.metadata.get("portrait_emb")
            if isinstance(portrait_path, str) and portrait_path:
                item.setData(0, Qt.UserRole + 1, portrait_path)
                item.setSizeHint(0, QSize(0, 54))
                self._portrait_queue.append((item, Path(portrait_path), generation))
            parent.addChild(item)
        for parent in groups.values():
            parent.setExpanded(True)
        self.tree.collapseAll()
        self.apply_filter(self.search.text())
        if self._portrait_queue:
            self._portrait_batch_timer.start(0)

    def _process_portrait_batch(self) -> None:
        """Decode a few portrait EMBs without freezing the Qt event loop."""
        generation = self._portrait_generation
        processed = 0
        while self._portrait_queue and processed < 2:
            item, path, queued_generation = self._portrait_queue.pop(0)
            if queued_generation != generation:
                continue
            # Items may have been destroyed by a refresh before this timer fires.
            try:
                icon = self._portrait_icon(path)
                if not icon.isNull():
                    item.setIcon(0, icon)
            except RuntimeError:
                pass
            processed += 1
        if self._portrait_queue:
            self._portrait_batch_timer.start(15)

    def apply_filter(self, text: str) -> None:
        query = text.strip().casefold()
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            visible = 0
            for j in range(parent.childCount()):
                child = parent.child(j)
                index = child.data(0, Qt.UserRole)
                entry = self.entries[index]
                haystack = f"{entry.label} {entry.path}".casefold()
                hidden = bool(query and query not in haystack)
                child.setHidden(hidden)
                visible += not hidden
            parent.setHidden(visible == 0)

    def library_selection_changed(self) -> None:
        selected = self.tree.selectedItems()
        if not selected:
            return
        index = selected[0].data(0, Qt.UserRole)
        if index is None:
            return
        entry = self.entries[index]
        if entry.category.startswith("Skills"):
            self.append_skill_entry(entry)
            return
        if entry.category.startswith("Sounds") or bool(entry.metadata.get("sound_bank")):
            self.load_sound_entry(entry)
            return
        self.show_entry(entry)

    def _portrait_icon(self, emb_path: Path) -> QIcon:
        key = str(emb_path.resolve())
        cached = self.portrait_cache.get(key)
        if cached is not None:
            return cached
        icon = QIcon()
        try:
            emb = load_emb(emb_path)
            candidates = []
            for embedded in emb.entries:
                image = embedded.to_qimage()
                if image is None or image.width() <= 0 or image.height() <= 0:
                    continue
                area = image.width() * image.height()
                square_bias = min(image.width(), image.height()) / max(image.width(), image.height())
                candidates.append((area * (0.65 + 0.35 * square_bias), image))
            if candidates:
                image = max(candidates, key=lambda pair: pair[0])[1]
                # Retain the original/highest-resolution pixmap. QIcon selects
                # its appropriate mip/downscale at paint time, including HiDPI.
                # Pre-shrinking to 40px blurred portraits on 125%-200% displays.
                pixmap = QPixmap.fromImage(image)
                icon = QIcon(pixmap)
        except (OSError, EmbFormatError):
            pass
        self.portrait_cache[key] = icon
        return icon

    def _clear_sound_state(self, clear_list: bool = True) -> None:
        """Release audio/metadata state held by the previously opened bank."""
        self._sound_metadata_generation += 1
        self._sound_metadata_timer.stop()
        self._sound_metadata_index = 0
        self.sound_player.stop()
        self.sound_player.setSource(QUrl())
        self.current_sound_bank = None
        self.current_sound_entry = None
        if clear_list:
            self.sound_list.clear()
            self.sound_meta.setText("Select an ACB sound bank from the library.")
        for button in (self.sound_play_button, self.sound_stop_button, self.sound_extract_button, self.sound_extract_all_button):
            button.setEnabled(False)

    def _release_loaded_asset_state(self) -> None:
        """Drop parsed files, previews and GPU-backed viewport data from the prior asset."""
        self.animation_timer.stop()
        self.current_ean = None
        self.face_ean = None
        self.lip_ean = None
        self.camera_ean = None
        self.current_animation = None
        self.current_emm = None
        self.current_emb = None
        self.animation_sources.clear()
        self.texture_preview.setPixmap(QPixmap())
        self.texture_preview.setText("No texture selected")
        self.texture_list.clear()
        self.material_tree.clear()
        self.parameter_table.setRowCount(0)
        self.skeleton_tree.clear()
        self.viewport.clear()
        # Qt/Python may otherwise wait for a later collection cycle before
        # releasing large QImages and parser object graphs.
        gc.collect()

    def load_sound_entry(self, entry: LibraryEntry) -> None:
        self._clear_sound_state(clear_list=True)
        acb_text = entry.metadata.get("acb_path")
        acb_path = Path(acb_text) if isinstance(acb_text, str) and acb_text else next((a.path for a in entry.assets if a.kind == AssetKind.SOUND_BANK), None)
        if acb_path is None:
            return
        self.current_sound_entry = entry
        self.sound_meta.setText(f"Reading {acb_path.name}…")
        self.tabs.setCurrentWidget(self.sound_tab)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            bank = inspect_sound_bank(acb_path, self.project_root)
        except (OSError, SoundToolError) as exc:
            self.sound_meta.setText(f"Could not open {acb_path.name}: {exc}")
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.current_sound_bank = bank
        self._sound_metadata_generation += 1
        self._sound_metadata_index = 0
        self._sound_metadata_timer.stop()
        for stream in bank.streams:
            item = QListWidgetItem(f"{stream.index:04d}  {stream.name}")
            item.setData(Qt.UserRole, stream.index)
            self.sound_list.addItem(item)
        awb_text = bank.awb_path.name if bank.awb_path else "embedded/no matching AWB"
        self.sound_meta.setText(f"{bank.acb_path.name} • {len(bank.streams)} sounds • AWB: {awb_text} • Output: 44,100 Hz WAV")
        enabled = bool(bank.streams)
        for button in (self.sound_play_button, self.sound_extract_button, self.sound_extract_all_button):
            button.setEnabled(enabled)
        self.sound_stop_button.setEnabled(False)
        if enabled:
            # Metadata is resolved only for the selected sound (and during an
            # explicit Extract All). Spawning one vgmstream process per cue in
            # the background can consume enormous memory on large ACB/AWB banks.
            self.sound_list.setCurrentRow(0)

    def _load_next_sound_metadata(self) -> None:
        bank = self.current_sound_bank
        if bank is None or self._sound_metadata_index >= len(bank.streams):
            return
        row = self._sound_metadata_index
        self._sound_metadata_index += 1
        stream = bank.streams[row]
        try:
            detailed = read_stream_metadata(bank, stream)
            bank.streams[row] = detailed
            item = self.sound_list.item(row)
            if item is not None:
                item.setText(f"{detailed.index:04d}  {detailed.name}")
        except (OSError, SoundToolError):
            pass
        if self.current_sound_bank is bank and self._sound_metadata_index < len(bank.streams):
            self._sound_metadata_timer.start(0)

    def _selected_sound_stream(self) -> SoundStream | None:
        bank = self.current_sound_bank
        row = self.sound_list.currentRow()
        if bank is None or not 0 <= row < len(bank.streams):
            return None
        return bank.streams[row]

    def sound_stream_selected(self, row: int) -> None:
        bank = self.current_sound_bank
        if bank is None or not 0 <= row < len(bank.streams):
            return
        stream = bank.streams[row]
        try:
            detailed = read_stream_metadata(bank, stream)
            bank.streams[row] = detailed
            item = self.sound_list.item(row)
            if item is not None:
                item.setText(f"{detailed.index:04d}  {detailed.name}")
            details = []
            if detailed.title:
                details.append(f"Title: {detailed.title}")
            if detailed.artist:
                details.append(f"Artist: {detailed.artist}")
            if detailed.sample_rate:
                details.append(f"{detailed.sample_rate:,} Hz")
            if detailed.channels:
                details.append(f"{detailed.channels} ch")
            if detailed.samples and detailed.sample_rate:
                details.append(f"{detailed.samples / detailed.sample_rate:.2f} sec")
            base = f"{bank.acb_path.name} • Sound {detailed.index}/{len(bank.streams)}"
            self.sound_meta.setText(base + (" • " + " • ".join(details) if details else ""))
        except (OSError, SoundToolError):
            pass

    def play_selected_sound(self) -> None:
        bank = self.current_sound_bank
        stream = self._selected_sound_stream()
        if bank is None or stream is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            wav_path = decode_stream_to_temp(bank, stream.index)
        except (OSError, SoundToolError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not decode sound:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.sound_player.stop()
        self.sound_player.setSource(QUrl())
        self.sound_player.setSource(QUrl.fromLocalFile(str(wav_path)))
        self.sound_player.play()
        self.sound_stop_button.setEnabled(True)
        self.statusBar().showMessage(f"Playing {stream.name}", 5000)

    def stop_sound(self) -> None:
        self.sound_player.stop()
        self.sound_player.setSource(QUrl())
        if hasattr(self, "sound_stop_button"):
            self.sound_stop_button.setEnabled(False)

    def extract_selected_sound(self) -> None:
        bank = self.current_sound_bank
        stream = self._selected_sound_stream()
        if bank is None or stream is None:
            return
        try:
            stream = read_stream_metadata(bank, stream)
            bank.streams[self.sound_list.currentRow()] = stream
        except (OSError, SoundToolError):
            pass
        default_name = f"{stream.index:04d}_{safe_stream_filename(stream)}.wav"
        output, _filter = QFileDialog.getSaveFileName(self, "Extract sound as WAV", default_name, "WAV audio (*.wav)")
        if not output:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            decode_stream(bank, stream.index, Path(output), 44100)
        except (OSError, SoundToolError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not extract sound:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Extracted {Path(output).name}", 7000)

    def extract_all_sounds(self) -> None:
        bank = self.current_sound_bank
        if bank is None:
            return
        output = QFileDialog.getExistingDirectory(self, "Choose sound extraction directory", str(bank.acb_path.parent))
        if not output:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            def report(current: int, total: int, name: str) -> None:
                self.statusBar().showMessage(f"Extracting {name} ({current}/{total})")
                QApplication.processEvents()
            files = extract_all_streams(bank, Path(output) / bank.acb_path.stem, report)
        except (OSError, SoundToolError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not extract bank:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        QMessageBox.information(self, APP_NAME, f"Extracted {len(files)} sounds as 44,100 Hz WAV files.")

    def show_entry(self, entry: LibraryEntry) -> None:
        self._clear_sound_state(clear_list=True)
        self._release_loaded_asset_state()
        self.current_entry = entry
        if entry.category in {"Characters", "CaC"}:
            self.current_character_entry = entry
            self.additional_animation_entries.clear()
            self.additional_animation_paths.clear()
            self.additional_camera_sources.clear()
            self.animation_sources.clear()
        self.character_title.setText(entry.label)
        status = "Playable / usable" if entry.usable is True else "Non-usable" if entry.usable is False else "Unknown availability"
        if entry.category == "CaC":
            profile = entry.metadata.get("cac_profile", {})
            race_name = entry.metadata.get("race_name", entry.metadata.get("race", "Unknown race"))
            level = profile.get("level", "?") if isinstance(profile, dict) else "?"
            preset = entry.metadata.get("preset_index", 0)
            self.character_meta.setText(
                f"CaC • {race_name} • Level {level} • Preset {int(preset) + 1} • {summarize_assets(entry.assets)}\n{entry.path}"
            )
        else:
            self.character_meta.setText(f"{entry.category} • {status} • {summarize_assets(entry.assets)}\n{entry.path}")
        self.asset_list.clear()
        self.animation_timer.stop()
        self.current_ean = None
        self.face_ean = None
        self.lip_ean = None
        self.camera_ean = None
        self.current_animation = None
        self.animation_list.clear()
        self.animation_sources.clear()
        for combo in (self.face_combo, self.lip_combo, self.camera_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("None", -1)
            combo.blockSignals(False)
        self.overlay_meta.setText("Facial: none • Lip: none • Camera: none")
        self.frame_slider.setRange(0, 0)
        self.frame_label.setText("Frame 0 / 0")
        self.play_button.setText("Play")
        self.material_tree.clear()
        self.parameter_table.setRowCount(0)
        self.material_meta.setText("Select an EMM file to inspect its shaders, colors, and parameters.")
        self.current_emm = None
        self.model_list.blockSignals(True)
        self.model_list.clear()
        self.model_list.blockSignals(False)
        for asset in entry.assets:
            item = QListWidgetItem(f"{asset.kind.value:18}  {asset.path.name}")
            item.setToolTip(str(asset.path))
            item.setData(Qt.UserRole, str(asset.path))
            item.setData(Qt.UserRole + 1, asset.kind.value)
            self.asset_list.addItem(item)
            if asset.kind == AssetKind.MODEL:
                model_item = QListWidgetItem(asset.path.name)
                model_item.setFlags(model_item.flags() | Qt.ItemIsUserCheckable)
                model_item.setCheckState(Qt.Unchecked)
                model_item.setData(Qt.UserRole, str(asset.path))
                model_item.setToolTip("Check to parse and render this native EMD. Double-click to fit the camera.")
                self.model_list.addItem(model_item)
        self.populate_assignments(entry)
        self.extract_button.setEnabled(True)
        race = str(entry.metadata.get("race", entry.code))
        explicit_skeleton = str(entry.metadata.get("main_skeleton", ""))
        main_skeleton = Path(explicit_skeleton) if explicit_skeleton and Path(explicit_skeleton).is_file() else None
        if main_skeleton is None:
            main_skeleton = next((a.path for a in entry.assets if a.kind == AssetKind.SKELETON and a.path.stem.casefold() == f"{race}_000".casefold()), None)
        if main_skeleton:
            self.load_skeleton_to_viewport(main_skeleton, switch_tab=False)
        # Load the correct rig before any mesh, including attached SCD physics.
        # Default all available LOD0/body/face/secondary parts on; never LODs.
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if "lod" not in Path(item.data(Qt.UserRole)).stem.casefold():
                item.setCheckState(Qt.Checked)

        self.viewport.fit_scene()

        explicit_animation = str(entry.metadata.get("main_animation", ""))
        main_animation = Path(explicit_animation) if explicit_animation and Path(explicit_animation).is_file() else None
        if main_animation is None:
            main_animation = next((a.path for a in entry.assets if a.kind == AssetKind.ANIMATION and a.path.name.casefold() == f"{race}.ean".casefold()), None)
        if main_animation is None:
            main_animation = next((a.path for a in entry.assets if a.kind == AssetKind.ANIMATION), None)
        face_path = next((a.path for a in entry.assets if a.path.name.casefold() == f"{race}.fce.ean".casefold()), None)
        lip_path = next((a.path for a in entry.assets if a.path.name.casefold() == f"{race}.fce.lip.ean".casefold()), None)
        camera_path = next((a.path for a in entry.assets if a.path.name.casefold() == f"{race}.cam.ean".casefold()), None)
        self.face_ean = self._try_load_aux_ean(face_path)
        self.lip_ean = self._try_load_aux_ean(lip_path)
        self.camera_ean = self._try_load_aux_ean(camera_path)
        self.populate_aux_animation_combos()
        if main_animation:
            self.load_animation_file(main_animation, switch_tab=False)

        if entry.category == "Characters":
            # Load skill EANs/cameras referencing the character ID (rather than
            # requiring the user to pick each matching skill by hand).
            matched = self._character_skill_index.get(entry.code.upper(), [])
            seen: set[Path] = set()
            self.additional_animation_entries = []
            for skill in matched:
                resolved = skill.path.resolve()
                if resolved not in seen:
                    self.additional_animation_entries.append(skill)
                    seen.add(resolved)
            self.additional_animation_paths = {
                a.path.resolve() for sk in self.additional_animation_entries
                for a in sk.assets if a.kind in (AssetKind.ANIMATION, AssetKind.CAMERA)
            }
            self._load_extra_camera_sources()
            self._rebuild_animation_list(select_first=False)
            self._rebuild_camera_combo()
        if entry.category == "CaC":
            skill_entries = resolve_skill_entries(entry.metadata_to_profile() if hasattr(entry, "metadata_to_profile") else self._profile_for_cac_entry(entry), self.entries, self.id_lists)
            self.additional_animation_entries = skill_entries
            self.additional_animation_paths = {asset.path.resolve() for skill in skill_entries for asset in skill.assets if asset.kind in (AssetKind.ANIMATION, AssetKind.CAMERA)}
            self._load_extra_camera_sources()
            self._rebuild_animation_list(select_first=False)
            self._rebuild_camera_combo()

    def _load_cached_ean(self, path: Path):
        key = path.resolve()
        ean = self._ean_cache.get(key)
        if ean is None:
            ean = load_ean(path)
            # A bounded cache keeps frequent character switches responsive without
            # retaining every skill EAN in a large extracted game indefinitely.
            if len(self._ean_cache) >= 16:
                self._ean_cache.pop(next(iter(self._ean_cache)))
            self._ean_cache[key] = ean
        return ean

    def _load_extra_camera_sources(self) -> None:
        self.additional_camera_sources.clear()
        for skill in self.additional_animation_entries:
            for asset in skill.assets:
                if asset.kind != AssetKind.CAMERA:
                    continue
                try:
                    ean = self._load_cached_ean(asset.path)
                except (OSError, EanFormatError):
                    continue
                for animation in ean.animations:
                    self.additional_camera_sources.append((ean, animation, skill.code, asset.path))

    def _profile_for_cac_entry(self, entry: LibraryEntry):
        slot = int(entry.metadata.get("cac_slot", -1))
        return next((profile for profile in self.cac_profiles if profile.slot == slot), self.cac_profiles[0] if self.cac_profiles else None)

    def _rebuild_animation_list(self, select_first: bool = False) -> None:
        """Rebuild the visible animation list from the character and appended skills."""
        self.animation_sources.clear()
        self.animation_list.blockSignals(True)
        self.animation_list.clear()

        if self.current_ean is not None:
            for animation in self.current_ean.animations:
                self.animation_sources.append((self.current_ean, animation, self.current_ean.path.name, self.current_ean.path, False))
                self.animation_list.addItem(
                    f"{animation.index:03d}  {animation.name}  ({animation.duration}f, {len(animation.nodes)} bones)"
                )

        for entry in self.additional_animation_entries:
            for asset in entry.assets:
                if asset.kind != AssetKind.ANIMATION:
                    continue
                try:
                    ean = self._load_cached_ean(asset.path)
                except (OSError, EanFormatError):
                    continue
                source_label = f"{entry.code} / {asset.path.name}"
                for animation in ean.animations:
                    self.animation_sources.append((ean, animation, source_label, asset.path, True))
                    self.animation_list.addItem(
                        f"[SKILL {entry.code}] {animation.index:03d}  {animation.name}  "
                        f"({animation.duration}f, {len(animation.nodes)} bones)"
                    )

        self.animation_list.blockSignals(False)
        self.remove_additional_button.setEnabled(bool(self.additional_animation_entries))
        additional_count = sum(1 for source in self.animation_sources if source[4])
        if self.current_ean is not None:
            self.animation_meta.setText(
                f"{self.current_ean.path.name} • {len(self.current_ean.animations)} character animations • "
                f"{additional_count} additional animations"
            )
        if select_first and self.animation_list.count():
            self.animation_list.setCurrentRow(0)

    def append_skill_entry(self, entry: LibraryEntry) -> None:
        """Append body and camera EANs from a skill without replacing prior sets."""
        if self.current_character_entry is None or self.current_entry is None or self.current_entry.category not in {"Characters", "CaC"}:
            QMessageBox.information(self, APP_NAME, "Load a character before appending skill animations.")
            return

        ean_assets = [asset for asset in entry.assets if asset.kind in (AssetKind.ANIMATION, AssetKind.CAMERA)]
        if not ean_assets:
            self.statusBar().showMessage(f"{entry.code} contains no EAN animations.", 6000)
            return

        new_assets = [asset for asset in ean_assets if asset.path.resolve() not in self.additional_animation_paths]
        if not new_assets:
            self.statusBar().showMessage(f"{entry.code} is already loaded.", 5000)
            return

        body_count = 0
        camera_count = 0
        valid_paths: list[Path] = []
        new_camera_sources: list[tuple[object, object, str, Path]] = []
        for asset in new_assets:
            try:
                ean = load_ean(asset.path)
            except (OSError, EanFormatError) as exc:
                self.statusBar().showMessage(f"Could not load {asset.path.name}: {exc}", 7000)
                continue
            valid_paths.append(asset.path.resolve())
            if getattr(ean, "is_camera", False) or asset.kind == AssetKind.CAMERA or ".cam." in asset.path.name.casefold():
                for animation in ean.animations:
                    new_camera_sources.append((ean, animation, entry.code, asset.path))
                camera_count += len(ean.animations)
            else:
                body_count += len(ean.animations)

        if not valid_paths:
            return

        self.additional_animation_entries.append(entry)
        self.additional_animation_paths.update(valid_paths)
        self.additional_camera_sources.extend(new_camera_sources)
        self._rebuild_animation_list(select_first=False)
        self._rebuild_camera_combo()
        self.tabs.setCurrentWidget(self.animation_tab)
        self.statusBar().showMessage(
            f"Appended {body_count} body and {camera_count} camera animations from {entry.label}. "
            f"Total additional sets: {len(self.additional_animation_entries)}",
            7000,
        )

    def remove_additional_animations(self) -> None:
        if not self.additional_animation_entries:
            return
        self.animation_timer.stop()
        self.current_animation = None
        self.viewport.set_animation(None)
        self.additional_animation_entries.clear()
        self.additional_animation_paths.clear()
        self.additional_camera_sources.clear()
        self._rebuild_animation_list(select_first=False)
        self._rebuild_camera_combo()
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_label.setText("Frame 0 / 0")
        self.play_button.setText("Play")
        self.statusBar().showMessage("Removed all additional skill animations.", 5000)

    def populate_assignments(self, entry: LibraryEntry) -> None:
        models = [a for a in entry.assets if a.kind == AssetKind.MODEL]
        skeletons = [a for a in entry.assets if a.kind == AssetKind.SKELETON]
        main = next((s.path for s in skeletons if s.path.stem.casefold() == f"{entry.code}_000".casefold()), None)
        if main is None:
            main = next((s.path for s in skeletons if s.path.stem.casefold() == entry.code.casefold()), None)
        self.assignment_table.setRowCount(len(models))
        for row, model in enumerate(models):
            assigned = model.paired_skeleton or main
            role = "Physics / secondary rig" if model.is_physics else "Main character rig"
            values = [model.path.name, assigned.name if assigned else "Unresolved", role]
            for col, value in enumerate(values):
                self.assignment_table.setItem(row, col, QTableWidgetItem(value))


    def asset_selected(self) -> None:
        selected = self.asset_list.selectedItems()
        if not selected:
            return
        item = selected[0]
        kind = item.data(Qt.UserRole + 1)
        path = Path(item.data(Qt.UserRole))
        if kind == AssetKind.SKELETON.value:
            self.load_skeleton_inspector(path)
        elif kind == AssetKind.MODEL.value:
            self.load_model_to_viewport(path)
        elif kind == AssetKind.MATERIAL.value:
            self.load_material_inspector(path)
        elif kind == AssetKind.TEXTURE.value and path.suffix.casefold() == ".emb":
            self.load_texture_inspector(path)
        elif kind == AssetKind.ANIMATION.value:
            self.load_animation_file(path, switch_tab=True)

    def load_material_inspector(self, path: Path) -> None:
        self.material_tree.clear()
        self.parameter_table.setRowCount(0)
        try:
            emm = load_emm(path)
        except (OSError, EmmFormatError) as exc:
            self.material_meta.setText(f"Failed to parse {path.name}: {exc}")
            self.tabs.setCurrentWidget(self.material_tab)
            return

        self.current_emm = emm
        endian_name = "Little endian" if emm.endian == "<" else "Big endian"
        version = ".".join(str(part) for part in emm.version)
        self.material_meta.setText(
            f"{path.name} • {len(emm.materials)} materials • {emm.parameter_count} parameters • "
            f"{endian_name} • Version {version}"
        )
        for index, material in enumerate(emm.materials):
            item = QTreeWidgetItem([material.name, material.shader, str(len(material.parameters))])
            item.setData(0, Qt.UserRole, index)
            color = material.preview_color()
            rgb = tuple(int(component * 255) for component in color[:3])
            item.setForeground(0, QBrush(QColor(*rgb)))
            item.setToolTip(0, f"Shader: {material.shader}\nUnknown: 0x{material.unknown:04X}")
            self.material_tree.addTopLevelItem(item)
        if self.material_tree.topLevelItemCount():
            self.material_tree.setCurrentItem(self.material_tree.topLevelItem(0))
        self.tabs.setCurrentWidget(self.material_tab)

        model_path = path.with_suffix(".emd")
        if str(model_path) in self.viewport.parts:
            self.viewport.apply_emm(str(model_path), emm)

    def material_selection_changed(self) -> None:
        selected = self.material_tree.selectedItems()
        if not selected or self.current_emm is None:
            return
        index = selected[0].data(0, Qt.UserRole)
        if index is None or not 0 <= index < len(self.current_emm.materials):
            return
        material = self.current_emm.materials[index]
        self.parameter_table.setRowCount(len(material.parameters))
        for row, parameter in enumerate(material.parameters):
            value = f"{parameter.value:.7g}" if isinstance(parameter.value, float) else str(parameter.value)
            values = [parameter.name, parameter.type_name, str(parameter.level), value]
            for column, text in enumerate(values):
                self.parameter_table.setItem(row, column, QTableWidgetItem(text))

        color = material.preview_color()
        r, g, b = (int(component * 255) for component in color[:3])
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        foreground = "#111111" if luminance > 145 else "#f5f5f5"
        self.material_swatch.setText(f"{material.name}  •  {material.shader}  •  RGB {r}, {g}, {b}")
        self.material_swatch.setStyleSheet(
            f"background: rgb({r}, {g}, {b}); color: {foreground}; "
            "border: 1px solid #646b75; border-radius: 5px; font-weight: 600;"
        )

    def load_texture_inspector(self, path: Path) -> None:
        self.texture_list.clear()
        self.texture_preview.setText("No texture selected")
        self.texture_preview.setPixmap(QPixmap())
        try:
            emb = load_emb(path)
        except (OSError, EmbFormatError) as exc:
            self.texture_meta.setText(f"Failed to parse {path.name}: {exc}")
            self.tabs.setCurrentWidget(self.texture_tab)
            return
        self.current_emb = emb
        valid = 0
        for entry in emb.entries:
            image = entry.to_qimage()
            text = f"{entry.index:03d}  {entry.name}"
            if image is not None:
                text += f"  ({image.width()}×{image.height()})"
                valid += 1
            else:
                text += "  [unsupported/non-image]"
            self.texture_list.addItem(text)
        version = ".".join(str(part) for part in emb.version)
        self.texture_meta.setText(f"{path.name} • {len(emb.entries)} entries • {valid} decoded textures • Version {version}")
        if self.texture_list.count():
            self.texture_list.setCurrentRow(0)
        self.tabs.setCurrentWidget(self.texture_tab)

    def texture_selection_changed(self, row: int) -> None:
        if self.current_emb is None or not 0 <= row < len(self.current_emb.entries):
            return
        entry = self.current_emb.entries[row]
        image = entry.to_qimage()
        if image is None:
            self.texture_preview.setPixmap(QPixmap())
            self.texture_preview.setText(f"{entry.name} could not be decoded as an image.")
            return
        pixmap = QPixmap.fromImage(image)
        self.texture_preview.setPixmap(pixmap.scaled(self.texture_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.texture_preview.setToolTip(f"{entry.name}\n{image.width()} × {image.height()}")

    def load_skeleton_inspector(self, path: Path) -> None:
        self.skeleton_tree.clear()
        try:
            esk = load_esk(path)
        except (OSError, EskFormatError) as exc:
            self.skeleton_meta.setText(f"Failed to parse {path.name}: {exc}")
            self.tabs.setCurrentWidget(self.skeleton_tab)
            return

        endian_name = "Little endian" if esk.endian == "<" else "Big endian"
        version = ".".join(str(part) for part in esk.version)
        self.skeleton_meta.setText(
            f"{path.name} • {len(esk.bones)} bones • {endian_name} • Version {version} • "
            f"Skeleton ID 0x{esk.skeleton_id:016X}"
        )

        items: dict[int, QTreeWidgetItem] = {}
        for bone in esk.bones:
            parent_name = "—" if bone.parent is None else esk.bones[bone.parent].name if bone.parent < len(esk.bones) else str(bone.parent)
            tree_item = QTreeWidgetItem([bone.name, str(bone.index), parent_name, f"0x{bone.ik_flag:04X}"])
            tree_item.setToolTip(0, 
                f"Position: {tuple(round(v, 5) for v in bone.position)}\n"
                f"Rotation: {tuple(round(v, 5) for v in bone.rotation)}\n"
                f"Scale: {tuple(round(v, 5) for v in bone.scale)}"
            )
            items[bone.index] = tree_item

        for bone in esk.bones:
            item = items[bone.index]
            if bone.parent is None or bone.parent not in items:
                self.skeleton_tree.addTopLevelItem(item)
            else:
                items[bone.parent].addChild(item)

        self.skeleton_tree.expandToDepth(1)
        self.viewport.set_esk(esk)
        self.tabs.setCurrentWidget(self.skeleton_tab)

    def _set_viewport_option(self, name: str, value: bool) -> None:
        setattr(self.viewport, name, value)
        self.viewport._invalidate_frame()

    def clear_viewport(self) -> None:
        self.model_list.blockSignals(True)
        for i in range(self.model_list.count()):
            self.model_list.item(i).setCheckState(Qt.Unchecked)
        self.model_list.blockSignals(False)
        self.viewport.clear()

    def load_model_to_viewport(self, path: Path) -> None:
        key = str(path)
        if key in self.viewport.parts:
            self.viewport.set_part_visible(key, True)
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            emd = load_emd(path)
        except (OSError, EmdFormatError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Failed to parse {path.name}:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        emm = None
        emm_path = path.with_suffix(".emm")
        if emm_path.is_file():
            try:
                emm = load_emm(emm_path)
            except (OSError, EmmFormatError) as exc:
                self.statusBar().showMessage(f"Could not apply {emm_path.name}: {exc}", 7000)
        emb = None
        emb_path = path.with_suffix(".emb")
        if emb_path.is_file():
            try:
                emb = load_emb(emb_path)
            except (OSError, EmbFormatError) as exc:
                self.statusBar().showMessage(f"Could not load {emb_path.name}: {exc}", 7000)
        dyt = None
        dyt_path = path.with_suffix(".dyt.emb")
        if dyt_path.is_file():
            try:
                dyt = load_emb(dyt_path)
            except (OSError, EmbFormatError) as exc:
                self.statusBar().showMessage(f"Could not load {dyt_path.name}: {exc}", 7000)
        secondary = None
        if "_scd" in path.stem.casefold():
            secondary_path = path.with_suffix(".esk")
            if not secondary_path.is_file():
                secondary_path = path.with_name(re.sub(r"_lod\d+$", "", path.stem, flags=re.I) + ".esk")
            if secondary_path.is_file():
                try:
                    secondary = load_esk(secondary_path)
                except (OSError, EskFormatError) as exc:
                    self.statusBar().showMessage(f"Physics skeleton error {secondary_path.name}: {exc}", 7000)
        self.viewport.set_emd(emd, key, emm, emb, dyt, secondary_esk=secondary)
        version = ".".join(str(part) for part in emd.version)
        mesh_count = sum(len(model.meshes) for model in emd.models)
        submesh_count = len(emd.submeshes)
        self.model_meta.setText(
            f"{path.name} • Version {version} • {len(emd.models)} model(s) • {mesh_count} mesh(es) • "
            f"{submesh_count} submesh(es) • {emd.vertex_count:,} vertices • {emd.triangle_count:,} triangles"
        )

    def load_skeleton_to_viewport(self, path: Path, switch_tab: bool = False) -> None:
        try:
            esk = load_esk(path)
        except (OSError, EskFormatError) as exc:
            self.statusBar().showMessage(f"Failed to parse {path.name}: {exc}", 7000)
            return
        if self.current_entry is not None and self.current_entry.category in {"Characters", "CaC"}:
            # Same merging strategy as the already-working SMD exporter.
            from .source_export import _build_combined_skeleton
            secondary = []
            for asset in self.current_entry.assets:
                if (asset.kind == AssetKind.SKELETON and "_scd" in asset.path.stem.casefold()
                        and "lod" not in asset.path.stem.casefold()):
                    try:
                        secondary.append(load_esk(asset.path))
                    except (OSError, EskFormatError):
                        pass
            if secondary:
                esk, _maps = _build_combined_skeleton(esk, secondary)
        self.viewport.set_esk(esk)
        if switch_tab:
            self.load_skeleton_inspector(path)

    def model_visibility_changed(self, item: QListWidgetItem) -> None:
        path = Path(item.data(Qt.UserRole))
        key = str(path)
        if item.checkState() == Qt.Checked:
            self.load_model_to_viewport(path)
        else:
            self.viewport.remove_part(key)

    def focus_model_item(self, item: QListWidgetItem) -> None:
        if item.checkState() != Qt.Checked:
            item.setCheckState(Qt.Checked)
        self.viewport.fit_scene()

    def set_all_models_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.model_list.count()):
            self.model_list.item(i).setCheckState(state)
        self.viewport.fit_scene()

    def load_animation_file(self, path: Path, switch_tab: bool = True) -> None:
        self.animation_timer.stop()
        self.play_button.setText("Play")
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            ean = self._load_cached_ean(path)
        except (OSError, EanFormatError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Failed to parse {path.name}:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        lower_name = path.name.casefold()
        if lower_name.endswith(".fce.lip.ean"):
            self.lip_ean = ean
            self._populate_animation_combo(self.lip_combo, self.lip_ean)
            self.refresh_animation_overlays()
            self.statusBar().showMessage(f"Loaded lip EAN: {path.name}", 5000)
            if switch_tab:
                self.tabs.setCurrentWidget(self.animation_tab)
            return
        if lower_name.endswith(".fce.ean"):
            self.face_ean = ean
            self._populate_animation_combo(self.face_combo, self.face_ean)
            self.refresh_animation_overlays()
            self.statusBar().showMessage(f"Loaded facial EAN: {path.name}", 5000)
            if switch_tab:
                self.tabs.setCurrentWidget(self.animation_tab)
            return
        if ean.is_camera:
            self.camera_ean = ean
            self._rebuild_camera_combo()
            self.refresh_animation_overlays()
            self.statusBar().showMessage(f"Loaded camera EAN: {path.name}", 5000)
            if switch_tab:
                self.tabs.setCurrentWidget(self.animation_tab)
            return
        self.current_ean = ean
        self.current_animation = None
        # Loading a character should not silently start a 60 FPS animation.
        # Manual animation clicks still auto-play through animation_selection_changed().
        self._rebuild_animation_list(select_first=switch_tab)
        version = ".".join(str(part) for part in ean.version)
        self.animation_meta.setText(
            f"{path.name} • {len(ean.animations)} base animations • Version {version} • 60 FPS native playback"
        )
        if switch_tab:
            self.tabs.setCurrentWidget(self.animation_tab)

    @staticmethod
    def _populate_animation_combo(combo: QComboBox, ean) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None", -1)
        if ean is not None:
            for index, animation in enumerate(ean.animations):
                combo.addItem(
                    f"{animation.index:03d}  {animation.name}  ({animation.duration}f)",
                    index,
                )
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def populate_aux_animation_combos(self) -> None:
        """Populate auxiliary selectors without unsafe auto-matching.

        Face and lip selectors store integer indices into their owning EAN. Camera
        selectors must store the animation object itself because the same combo also
        contains camera clips appended from multiple skill EAN files.
        """
        self._populate_animation_combo(self.face_combo, self.face_ean)
        self._populate_animation_combo(self.lip_combo, self.lip_ean)
        self._rebuild_camera_combo()

    @staticmethod
    def _selected_aux_animation(combo: QComboBox, ean):
        if ean is None:
            return None
        index = combo.currentData()
        if not isinstance(index, int) or index < 0 or index >= len(ean.animations):
            return None
        return ean.animations[index]

    def _rebuild_camera_combo(self) -> None:
        """Combine the character camera EAN with every appended skill camera EAN."""
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem("None", None)
        if self.camera_ean is not None:
            for animation in self.camera_ean.animations:
                self.camera_combo.addItem(
                    f"[CHARACTER] {animation.index:03d}  {animation.name}  ({animation.duration}f)",
                    animation,
                )
        for ean, animation, source_label, source_path in self.additional_camera_sources:
            self.camera_combo.addItem(
                f"[SKILL {source_label}] {animation.index:03d}  {animation.name}  ({animation.duration}f)",
                animation,
            )
        self.camera_combo.setCurrentIndex(0)
        self.camera_combo.blockSignals(False)

    def selected_face_animation(self):
        return self._selected_aux_animation(self.face_combo, self.face_ean)

    def selected_lip_animation(self):
        return self._selected_aux_animation(self.lip_combo, self.lip_ean)

    def selected_camera_animation(self):
        data = self.camera_combo.currentData()

        # Current camera-combo entries store the EanAnimation object directly so
        # character and appended skill cameras can coexist. Retain integer support
        # for any stale combo state created by an older build.
        if hasattr(data, "name"):
            return data
        if (
            isinstance(data, int)
            and self.camera_ean is not None
            and 0 <= data < len(self.camera_ean.animations)
        ):
            return self.camera_ean.animations[data]
        return None

    def _try_load_aux_ean(self, path: Path | None):
        if path is None:
            return None
        try:
            return self._load_cached_ean(path)
        except (OSError, EanFormatError):
            return None

    @staticmethod
    def _exact_matching_animation(ean, body_animation):
        """Return an auxiliary animation only when its name genuinely matches.

        Facial and lip EANs are libraries of one-frame poses, not parallel arrays
        aligned to the body EAN. Falling back to the body animation index applies
        unrelated poses and corrupts the character, so index fallback is forbidden.
        """
        if ean is None or body_animation is None:
            return None
        wanted = body_animation.name.casefold()
        return next((anim for anim in ean.animations if anim.name.casefold() == wanted), None)

    @staticmethod
    def _matching_camera_animation(ean, body_animation):
        """Match Xenoverse camera clips to their corresponding body clips.

        Cinematic body clips commonly use names such as ``CL3_DEM_START`` while
        their camera clip is named ``CL3_CAM_START``. Camera EAN indices are not
        aligned with body EAN indices, so matching is name-based only.
        """
        if ean is None or body_animation is None:
            return None

        body_name = body_animation.name.casefold()
        by_name = {anim.name.casefold(): anim for anim in ean.animations}

        # Some files do use identical names.
        exact = by_name.get(body_name)
        if exact is not None:
            return exact

        candidates: list[str] = []
        replacements = (
            ("_dem_", "_cam_"),
            ("_demo_", "_cam_"),
            ("_cut_", "_cam_"),
            ("_evt_", "_cam_"),
        )
        for source, target in replacements:
            if source in body_name:
                candidates.append(body_name.replace(source, target, 1))

        # Also compare the semantic suffix after DEM/CAM. This handles files whose
        # character prefix differs while still avoiding unsafe index matching.
        marker_suffix = None
        for marker in ("_dem_", "_demo_", "_cut_", "_evt_"):
            if marker in body_name:
                marker_suffix = body_name.split(marker, 1)[1]
                break
        if marker_suffix:
            for animation in ean.animations:
                camera_name = animation.name.casefold()
                if "_cam_" in camera_name and camera_name.split("_cam_", 1)[1] == marker_suffix:
                    return animation

        for candidate in candidates:
            match = by_name.get(candidate)
            if match is not None:
                return match
        return None

    def refresh_animation_overlays(self) -> None:
        face = self.selected_face_animation()
        lip = self.selected_lip_animation()
        camera = self.selected_camera_animation()
        self.viewport.set_overlay_animations(face, lip)
        self.viewport.set_camera_animation(camera, self.camera_box.isChecked() and camera is not None)
        self.overlay_meta.setText(
            f"Facial: {face.name if face else 'none selected'} • "
            f"Lip: {lip.name if lip else 'none selected'} • "
            f"Camera: {camera.name if camera else 'none selected'}"
        )

    def animation_selection_changed(self, row: int) -> None:
        if not 0 <= row < len(self.animation_sources):
            return
        self.animation_timer.stop()
        self.play_button.setText("Play")
        source_ean, animation, source_label, source_path, is_additional = self.animation_sources[row]
        self.current_animation = animation
        maximum = max(0, animation.duration - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, maximum)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"Frame 0 / {maximum}")
        self.viewport.set_animation(animation)
        # Pair body EANs with their exact face/lip/camera clips on selection;
        # camera EAN indices are NOT interchangeable with body EAN indices.
        for combo, ean in ((self.face_combo, self.face_ean), (self.lip_combo, self.lip_ean)):
            matched = self._exact_matching_animation(ean, animation)
            wanted = next((i for i, clip in enumerate(ean.animations) if clip is matched), -1) if ean is not None and matched else -1
            combo.blockSignals(True)
            combo.setCurrentIndex(wanted + 1 if wanted >= 0 else 0)
            combo.blockSignals(False)
        camera_matches = []
        if self.camera_ean is not None:
            clip = self._matching_camera_animation(self.camera_ean, animation)
            if clip is not None:
                camera_matches.append((self.camera_ean.path, clip))
        matched_camera_by_ean = {}
        for ean, clip, label, path in self.additional_camera_sources:
            key = id(ean)
            if key not in matched_camera_by_ean:
                matched_camera_by_ean[key] = self._matching_camera_animation(ean, animation)
            if clip is matched_camera_by_ean[key] and clip is not None:
                camera_matches.append((path, clip))
        # Prefer the camera alongside this body EAN, then the character camera.
        camera_matches.sort(key=lambda pair: (pair[0].parent != source_path.parent, pair[0] != source_path))
        camera_clip = camera_matches[0][1] if camera_matches else None
        self.camera_combo.blockSignals(True)
        camera_index = next((i for i in range(1, self.camera_combo.count())
                             if self.camera_combo.itemData(i) is camera_clip), 0) if camera_clip else 0
        self.camera_combo.setCurrentIndex(camera_index)
        self.camera_combo.blockSignals(False)
        self.refresh_animation_overlays()
        kind = "Additional / skill" if is_additional else "Character"
        self.animation_meta.setText(
            f"{kind}: {source_label} • {animation.name} • {animation.duration} frames • "
            f"{len(animation.nodes)} animated bones • {animation.frame_float_size * 16}-bit values"
        )
        # Clicking an animation immediately starts playback.
        self.animation_timer.start()
        self.play_button.setText("Pause")

    def animation_frame_changed(self, frame: int) -> None:
        if self.current_animation is None:
            return
        maximum = max(0, self.current_animation.duration - 1)
        self.frame_label.setText(f"Frame {frame} / {maximum}")
        self.viewport.set_animation_frame(float(frame))

    def toggle_animation_playback(self) -> None:
        if self.current_animation is None:
            return
        if self.animation_timer.isActive():
            self.animation_timer.stop()
            self.play_button.setText("Play")
        else:
            self.animation_timer.start()
            self.play_button.setText("Pause")

    def stop_animation(self) -> None:
        self.animation_timer.stop()
        self.play_button.setText("Play")
        self.frame_slider.setValue(0)

    def disable_animation(self) -> None:
        """Clear body/face/lip/camera animation and restore the bind/T-pose."""
        self.animation_timer.stop()
        self.play_button.setText("Play")
        self.current_animation = None
        self.animation_list.blockSignals(True)
        self.animation_list.clearSelection()
        self.animation_list.setCurrentRow(-1)
        self.animation_list.blockSignals(False)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText("Frame 0 / 0")
        for combo in (self.face_combo, self.lip_combo, self.camera_combo):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self.camera_box.setChecked(False)
        self.viewport.set_overlay_animations(None, None)
        self.viewport.set_camera_animation(None, False)
        self.viewport.set_animation(None)
        self.overlay_meta.setText("Facial: none • Lip: none • Camera: disabled")
        if self.current_ean is not None:
            self.animation_meta.setText(
                f"{self.current_ean.path.name} • Animation disabled • Bind/T-pose"
            )

    def advance_animation(self) -> None:
        if self.current_animation is None:
            self.animation_timer.stop()
            return
        maximum = max(0, self.current_animation.duration - 1)
        step = max(1, round(self.speed_spin.value()))
        next_frame = self.frame_slider.value() + step
        if next_frame > maximum:
            if self.loop_box.isChecked():
                next_frame = 0
            else:
                self.animation_timer.stop()
                self.play_button.setText("Play")
                next_frame = maximum
        self.frame_slider.setValue(next_frame)

    def extract_character(self) -> None:
        if self.current_entry is None:
            return
        default_root = self.settings.get("export_directory") or str(self.current_entry.path.parent / "_SourceExport")
        output_root = QFileDialog.getExistingDirectory(self, "Choose Source export directory", default_root)
        if not output_root:
            return
        self.settings["export_directory"] = output_root
        save_settings(self.settings)
        options = ExportOptions(
            meshes=self.meshes_box.isChecked(),
            animations=self.anim_box.isChecked(),
            textures=self.texture_box.isChecked(),
            events=self.event_box.isChecked(),
        )
        self.extract_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            def report(message: str, current: int, total: int) -> None:
                self.statusBar().showMessage(f"{message} ({current}/{total})")
                QApplication.processEvents()

            result = export_character(
                self.current_entry,
                output_root,
                options,
                report,
                additional_entries=self.additional_animation_entries,
            )
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"Extraction failed:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.extract_button.setEnabled(True)

        warning_text = ""
        if result.warnings:
            preview = "\n".join(f"• {warning}" for warning in result.warnings[:12])
            more = len(result.warnings) - 12
            warning_text = f"\n\nWarnings:\n{preview}" + (f"\n• ...and {more} more (see export_manifest.json)" if more > 0 else "")
        QMessageBox.information(
            self,
            APP_NAME,
            f"Source extraction complete.\n\n"
            f"Meshes: {len(result.mesh_files)}\n"
            f"Animations: {len(result.animation_files)}\n"
            f"Textures: {len(result.texture_files)}\n"
            f"Metadata: {len(result.metadata_files)}\n\n"
            f"Output: {result.output_dir}{warning_text}",
        )
        self.statusBar().showMessage(f"Exported to {result.output_dir}", 10000)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    project_root = Path(__file__).resolve().parent.parent
    window = MainWindow(project_root)
    window.show()
    QTimer.singleShot(250, window.ui_sounds.play_window_open)
    return app.exec()
