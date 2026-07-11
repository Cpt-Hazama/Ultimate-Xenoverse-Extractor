from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AssetKind(str, Enum):
    MODEL = "Model"
    SKELETON = "Skeleton"
    MATERIAL = "Material"
    TEXTURE = "Texture"
    ANIMATION = "Animation"
    CAMERA = "Camera Animation"
    FACE_ANIMATION = "Face Animation"
    LIP_ANIMATION = "Lip Animation"
    CONFIG = "Configuration"
    GAMEPLAY = "Gameplay Data"
    OTHER = "Other"


@dataclass(slots=True)
class IdRecord:
    numeric_id: str = ""
    hex_id: str = ""
    code: str = ""
    name: str = ""
    usable: bool | None = None
    notes: str = ""
    source: str = ""


@dataclass(slots=True)
class AssetFile:
    path: Path
    kind: AssetKind
    logical_name: str
    root_code: str
    is_physics: bool = False
    paired_skeleton: Path | None = None


@dataclass(slots=True)
class LibraryEntry:
    path: Path
    category: str
    code: str
    display_name: str
    usable: bool | None = None
    id_record: IdRecord | None = None
    assets: list[AssetFile] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.display_name} [{self.code}]" if self.display_name != self.code else self.code
