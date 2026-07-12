from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .id_lists import index_records
from .models import AssetFile, AssetKind, IdRecord, LibraryEntry
from .scanner import classify, logical_name
from .parsers.bcs import BcsFile, BcsFormatError, load_bcs

CAC_STORE_VERSION = 1

# Several CaC races reuse another race's visual-part files. In particular,
# Saiyans keep their animation/skeleton data in SYM/SYF while their body,
# face, hair and clothing models are shared with HUM/HUF.
MODEL_SOURCE_RACE = {
    "HUM": "HUM",
    "HUF": "HUF",
    "SYM": "HUM",
    "SYF": "HUF",
    "NMC": "NMC",
    "FRI": "FRI",
    "MAM": "MAM",
    "MAF": "MAF",
}

RACE_NAMES = {
    "HUM": "Earthling Male",
    "HUF": "Earthling Female",
    "SYM": "Saiyan Male",
    "SYF": "Saiyan Female",
    "NMC": "Namekian",
    "FRI": "Frieza Race",
    "MAM": "Majin Male",
    "MAF": "Majin Female",
}

_APPEARANCE_PARTS = {
    "FaceBase": "Face_base",
    "FaceForehead": "Face_forehead",
    "Eyes": "Face_eye",
    "Nose": "Face_nose",
    "Ears": "Face_ear",
    "Hair": "Hair",
}

_CLOTHING_PARTS = {
    "Top": "Bust",
    "Bottom": "Pants",
    "Gloves": "Rist",
    "Shoes": "Boots",
}

_SKILL_FIELDS = {
    "SuperSkill1": "Super IDs",
    "SuperSkill2": "Super IDs",
    "SuperSkill3": "Super IDs",
    "SuperSkill4": "Super IDs",
    "UltimateSkill1": "Ultimate IDs",
    "UltimateSkill2": "Ultimate IDs",
    "EvasiveSkill": "Evasive IDs",
    "BlastSkill": "Ki Blast IDs",
    "AwokenSkill": "Awoken IDs",
}


@dataclass(slots=True)
class CacPreset:
    index: int
    equipment: dict[str, int] = field(default_factory=dict)
    colors: dict[str, int] = field(default_factory=dict)
    skills: dict[str, int] = field(default_factory=dict)
    talisman: int = -1


@dataclass(slots=True)
class CacProfile:
    slot: int
    name: str
    race: str
    voice: int = 0
    level: int = 1
    current_preset: int = 0
    body_size: dict[str, int] = field(default_factory=dict)
    appearance: dict[str, int] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    presets: list[CacPreset] = field(default_factory=list)
    source_file: str = ""

    @property
    def race_name(self) -> str:
        return RACE_NAMES.get(self.race.upper(), self.race.upper())

    def active_preset(self) -> CacPreset:
        if not self.presets:
            return CacPreset(0)
        index = min(max(int(self.current_preset), 0), len(self.presets) - 1)
        return self.presets[index]


def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _node_value(parent: ET.Element, tag: str, default: int = 0) -> int:
    node = parent.find(tag)
    return _int(node.get("value") if node is not None else None, default)


def parse_save_xml(path: Path) -> list[CacProfile]:
    """Parse the decrypted XML representation produced by XV2 save editors.

    The XML can still say ``IsEncrypted=true`` because it retains the original
    save metadata. What matters is that the structured ``Characters/CaC`` nodes
    are present.
    """
    root = ET.parse(path).getroot()
    characters = root.find("Characters")
    if characters is None:
        raise ValueError("The XML does not contain a Characters section.")

    profiles: list[CacProfile] = []
    for slot, node in enumerate(characters.findall("CaC")):
        raw_name = (node.get("Name") or "").strip()
        if not raw_name:
            # Empty save slots are serialized as default HUM CaCs with no name.
            continue
        name = raw_name
        race_node = node.find("Race")
        race = ((race_node.get("value") if race_node is not None else "") or "").upper()
        if not race:
            continue

        appearance_node = node.find("Appearence") or node.find("Appearance")
        body_size: dict[str, int] = {}
        appearance: dict[str, int] = {}
        if appearance_node is not None:
            body = appearance_node.find("BodySize")
            if body is not None:
                body_size = {"height": _int(body.get("Height")), "width": _int(body.get("Width"))}
            for child in appearance_node:
                if child.tag == "BodySize":
                    continue
                if "value" in child.attrib:
                    appearance[child.tag] = _int(child.get("value"), -1)

        stats_node = node.find("Stats")
        stats = {key: _int(value) for key, value in (stats_node.attrib.items() if stats_node is not None else [])}

        presets: list[CacPreset] = []
        presets_node = node.find("Presets")
        if presets_node is not None:
            for preset_index, preset_node in enumerate(presets_node.findall("CacPreset")):
                equipment = {tag: _node_value(preset_node, tag, -1) for tag in (*_CLOTHING_PARTS.keys(), "Accessory")}
                colors = {
                    child.tag: _int(child.get("value"), 0)
                    for child in preset_node
                    if child.tag.endswith(("Color1", "Color2", "Color3", "Color4"))
                }
                skills: dict[str, int] = {}
                for field_name in _SKILL_FIELDS:
                    skill_node = preset_node.find(field_name)
                    skills[field_name] = _int(skill_node.get("ID1") if skill_node is not None else None, -1)
                presets.append(
                    CacPreset(
                        index=preset_index,
                        equipment=equipment,
                        colors=colors,
                        skills=skills,
                        talisman=_node_value(preset_node, "Talisman", -1),
                    )
                )

        profiles.append(
            CacProfile(
                slot=slot,
                name=name,
                race=race,
                voice=_node_value(node, "Voice", 0),
                level=_node_value(node, "Level", 1),
                current_preset=_node_value(node, "I_32", 0),
                body_size=body_size,
                appearance=appearance,
                stats=stats,
                presets=presets,
                source_file=str(path),
            )
        )
    return profiles


def save_profiles(path: Path, profiles: Iterable[CacProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CAC_STORE_VERSION,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": [
            {
                **asdict(profile),
                "presets": [asdict(preset) for preset in profile.presets],
            }
            for profile in profiles
        ],
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def load_profiles(path: Path) -> list[CacProfile]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    result: list[CacProfile] = []
    for raw in payload.get("profiles", []):
        try:
            presets = [CacPreset(**preset) for preset in raw.get("presets", [])]
            result.append(
                CacProfile(
                    slot=int(raw.get("slot", len(result))),
                    name=str(raw.get("name", f"CaC {len(result) + 1}")),
                    race=str(raw.get("race", "")).upper(),
                    voice=int(raw.get("voice", 0)),
                    level=int(raw.get("level", 1)),
                    current_preset=int(raw.get("current_preset", 0)),
                    body_size={str(k): int(v) for k, v in raw.get("body_size", {}).items()},
                    appearance={str(k): int(v) for k, v in raw.get("appearance", {}).items()},
                    stats={str(k): int(v) for k, v in raw.get("stats", {}).items()},
                    presets=presets,
                    source_file=str(raw.get("source_file", "")),
                )
            )
        except (TypeError, ValueError):
            continue
    return result


def _find_data_root(selected: Path) -> Path | None:
    selected = selected.resolve()
    if selected.name.casefold() == "data":
        return selected
    if (selected / "data").is_dir():
        return selected / "data"
    if selected.name.casefold() in {"chara", "skill", "stage", "item"} and selected.parent.name.casefold() == "data":
        return selected.parent
    # A direct chara selection is common during development.
    if selected.name.casefold() == "chara":
        return selected.parent
    return None


def _asset(path: Path, root_code: str, paired: Path | None = None) -> AssetFile:
    kind = classify(path)
    return AssetFile(
        path=path,
        kind=kind,
        logical_name=logical_name(path),
        root_code=root_code,
        is_physics=kind == AssetKind.MODEL and ("_scd" in path.stem.casefold() or paired is not None),
        paired_skeleton=paired,
    )


def _companions_for_stem(folder: Path, stem: str, root_code: str) -> list[AssetFile]:
    """Return one selected CaC part and all files needed to display it.

    A part normally has EMD/EMM/EMB/DYT companions. Hair, skirts and some
    accessories may additionally have a same-prefix ``_scd`` secondary model
    and ESK. LOD files are intentionally excluded from the automatic CaC set.
    """
    if not folder.is_dir():
        return []

    files: list[AssetFile] = []
    lower_stem = stem.casefold()

    def belongs(path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name.casefold()
        raw_stem = path.stem.casefold()
        if raw_stem == lower_stem:
            return True
        if name.startswith(lower_stem + ".dyt."):
            return True
        # Physics/secondary files belonging to the selected part.
        if raw_stem.startswith(lower_stem + "_scd") and "_lod" not in raw_stem:
            return True
        return False

    matches = [path for path in folder.iterdir() if belongs(path)]
    skeletons = {
        path.stem.casefold(): path
        for path in matches
        if path.suffix.casefold() == ".esk"
    }
    for path in sorted(matches, key=lambda item: item.name.casefold()):
        paired = None
        if path.suffix.casefold() == ".emd":
            paired = skeletons.get(path.stem.casefold())
        files.append(_asset(path, root_code, paired))
    return files


def _find_reference_assets(
    chara_root: Path,
    folder_codes: Iterable[str],
    root_code: str,
    stems: Iterable[str],
) -> list[AssetFile]:
    """Resolve BCS model/override references against extracted character folders."""
    assets: list[AssetFile] = []
    seen_stems: set[tuple[str, str]] = set()
    for folder_code in folder_codes:
        folder = chara_root / folder_code
        if not folder.is_dir():
            continue
        for raw_stem in stems:
            stem = Path(raw_stem.replace('\\', '/')).name
            lower = stem.casefold()
            for suffix in ('.dyt.emb', '.emd', '.emm', '.emb', '.esk', '.ean', '.fce'):
                if lower.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            if not stem or (folder_code.casefold(), stem.casefold()) in seen_stems:
                continue
            seen_stems.add((folder_code.casefold(), stem.casefold()))
            assets.extend(_companions_for_stem(folder, stem, root_code))
    return assets


def _exact_part_assets(
    chara_root: Path,
    folder_codes: Iterable[str],
    root_code: str,
    model_ids: Iterable[int],
    suffix: str,
) -> list[AssetFile]:
    """Resolve exact CaC part filenames without substring guessing."""
    for model_id in model_ids:
        if model_id < 0:
            continue
        for folder_code in folder_codes:
            folder = chara_root / folder_code
            if not folder.is_dir():
                continue
            stem = f"{folder_code}_{model_id:03d}_{suffix}"
            resolved = _companions_for_stem(folder, stem, root_code)
            if any(asset.kind == AssetKind.MODEL for asset in resolved):
                return resolved
    return []


def _part_model_assets(
    chara_root: Path,
    animation_race: str,
    model_race: str,
    part: object,
    suffix: str,
) -> list[AssetFile]:
    """Resolve the actual model for one BCS part.

    The four BCS string fields are per-file overrides; they are not four extra
    EMD models. Loading every string as a model stem caused the duplicate face
    pieces seen in the previous build. The numeric model field is the primary
    EMD selection, while explicit .emd overrides and physics references are
    handled separately.
    """
    assets: list[AssetFile] = []
    part_name = (getattr(part, "name", "") or model_race).strip("\x00 ") or model_race
    model_id = int(getattr(part, "model", 0xFFFF))
    if model_id != 0xFFFF:
        assets.extend(_exact_part_assets(
            chara_root,
            tuple(dict.fromkeys((part_name, model_race, animation_race))),
            animation_race,
            (model_id,),
            suffix,
        ))

    # Some modded BCS files explicitly override the EMD filename. Only accept
    # strings that actually point at an EMD (or whose exact stem contains the
    # expected slot suffix); do not treat EMM/EMB/FCE references as meshes.
    if not any(asset.kind == AssetKind.MODEL for asset in assets):
        explicit_stems: list[str] = []
        for raw in getattr(part, "files", ()):
            value = str(raw or "").replace("\\", "/").rsplit("/", 1)[-1]
            lower = value.casefold()
            if lower.endswith(".emd"):
                explicit_stems.append(value[:-4])
            elif suffix.casefold() in lower and not lower.endswith((".emm", ".emb", ".fce")):
                explicit_stems.append(Path(value).stem)
        assets.extend(_find_reference_assets(
            chara_root,
            tuple(dict.fromkeys((part_name, model_race, animation_race))),
            animation_race,
            explicit_stems,
        ))

    # Physics objects reference their own secondary model/material/texture set.
    physics_stems: list[str] = []
    for physics in getattr(part, "physics", ()):
        for raw in getattr(physics, "files", ()):
            value = str(raw or "").replace("\\", "/").rsplit("/", 1)[-1]
            lower = value.casefold()
            for ext in (".dyt.emb", ".emd", ".emm", ".emb", ".esk"):
                if lower.endswith(ext):
                    value = value[:-len(ext)]
                    break
            if value and value not in physics_stems:
                physics_stems.append(value)
    if physics_stems:
        assets.extend(_find_reference_assets(
            chara_root,
            tuple(dict.fromkeys((part_name, model_race, animation_race))),
            animation_race,
            physics_stems,
        ))
    return _dedupe_assets(assets)


def _resolve_bcs_part_assets(
    chara_root: Path,
    animation_race: str,
    model_race: str,
    bcs: BcsFile | None,
    set_index: int,
    slot: int,
    suffix: str,
) -> list[AssetFile]:
    """Resolve one CaC visual slot through BCS with exact safe fallbacks."""
    if set_index < 0:
        return []

    folder_codes = tuple(dict.fromkeys((model_race, animation_race)))
    candidates = [set_index]
    if set_index >= 1000:
        candidates.append(set_index - 1000)
    else:
        candidates.append(set_index + 1000)

    if bcs is not None:
        for candidate in candidates:
            part = bcs.get_part(candidate, slot)
            if part is None or part.model == 0xFFFF:
                continue
            resolved = _part_model_assets(
                chara_root, animation_race, model_race, part, suffix,
            )
            if any(asset.kind == AssetKind.MODEL for asset in resolved):
                return resolved

        # Save editors and modded inventories can expose a model ID rather than
        # the BCS part-set index. Search only the requested slot for an exact
        # model-ID match. This is deterministic and avoids old substring bugs.
        for wanted_model in candidates:
            matches = []
            for part_set in bcs.part_sets:
                if part_set is None:
                    continue
                part = part_set.part(slot)
                if part is not None and part.model == wanted_model:
                    matches.append(part)
            for part in matches:
                resolved = _part_model_assets(
                    chara_root, animation_race, model_race, part, suffix,
                )
                if any(asset.kind == AssetKind.MODEL for asset in resolved):
                    return resolved

    # Final exact-file fallback for assets not represented by the vanilla BCS
    # (common with modded hair/clothing). No partial or substring matching.
    return _exact_part_assets(
        chara_root, folder_codes, animation_race, candidates, suffix,
    )

def _part_stem(race: str, model_id: int, suffix: str) -> str:
    return f"{race}_{max(model_id, 0):03d}_{suffix}"


def _dedupe_assets(assets: Iterable[AssetFile]) -> list[AssetFile]:
    seen: set[Path] = set()
    result: list[AssetFile] = []
    for asset in assets:
        key = asset.path.resolve()
        if key in seen:
            continue
        seen.add(key)
        result.append(asset)
    return sorted(result, key=lambda asset: (asset.kind.value, asset.path.name.casefold()))


def _records_named(all_lists: dict[str, list[IdRecord]], name: str) -> list[IdRecord]:
    return next((records for list_name, records in all_lists.items() if list_name.casefold() == name.casefold()), [])


def resolve_skill_entries(profile: CacProfile, library_entries: list[LibraryEntry], all_lists: dict[str, list[IdRecord]]) -> list[LibraryEntry]:
    preset = profile.active_preset()
    wanted: list[tuple[str, int]] = [(_SKILL_FIELDS[field], skill_id) for field, skill_id in preset.skills.items() if skill_id >= 0]
    if not wanted:
        return []

    result: list[LibraryEntry] = []
    seen: set[Path] = set()
    for list_name, skill_id in wanted:
        lookup = index_records(_records_named(all_lists, list_name))
        record = lookup.get(str(skill_id))
        for entry in library_entries:
            if not entry.category.startswith("Skills"):
                continue
            match = False
            if entry.id_record is not None and record is not None:
                match = entry.id_record.numeric_id == record.numeric_id or entry.id_record.code == record.code
            if not match:
                prefix = entry.code.split("_", 1)[0]
                match = prefix.isdigit() and int(prefix) == skill_id
            if match and entry.path.resolve() not in seen:
                seen.add(entry.path.resolve())
                result.append(entry)
                break
    return result


def build_cac_entries(
    selected: Path,
    profiles: Iterable[CacProfile],
    all_lists: dict[str, list[IdRecord]],
) -> list[LibraryEntry]:
    data_root = _find_data_root(selected)
    if data_root is None:
        return []
    chara_root = data_root / "chara"
    if not chara_root.is_dir():
        return []

    entries: list[LibraryEntry] = []
    bcs_cache: dict[Path, BcsFile | None] = {}

    for profile in profiles:
        race = profile.race.upper()
        race_folder = chara_root / race
        model_race = MODEL_SOURCE_RACE.get(race, race)
        model_folder = chara_root / model_race
        if not race_folder.is_dir() or not model_folder.is_dir():
            continue

        # Saiyans share HUM/HUF visual BCS data; their own folders only provide
        # the race-specific skeleton and animation sets.
        bcs_path = model_folder / f"{model_race}.bcs"
        if bcs_path not in bcs_cache:
            try:
                bcs_cache[bcs_path] = load_bcs(bcs_path) if bcs_path.is_file() else None
            except (OSError, BcsFormatError):
                bcs_cache[bcs_path] = None
        visual_bcs = bcs_cache[bcs_path]

        assets: list[AssetFile] = []
        # Body animations remain race-specific (SYM/SYF), while Saiyan visual
        # rigs and face assets are shared with HUM/HUF. Load core files from
        # both folders without duplicating identical paths.
        race_core_names = {
            f"{race}_000.esk", f"{race}.esk", f"{race}.ean",
            f"{race}.fce.ean", f"{race}.fce.lip.ean", f"{race}.cam.ean",
        }
        model_core_names = {
            f"{model_race}_000.esk", f"{model_race}.esk",
            f"{model_race}.fce.ean", f"{model_race}.fce.lip.ean",
        }
        for folder, names in ((race_folder, race_core_names), (model_folder, model_core_names)):
            wanted = {name.casefold() for name in names}
            for path in folder.iterdir():
                if path.is_file() and path.name.casefold() in wanted:
                    assets.append(_asset(path, race))
        if bcs_path.is_file():
            assets.append(_asset(bcs_path, race))

        # BCS slot order: face base, forehead, eye, nose, ear, hair, bust,
        # pants, rist, boots. Save IDs select part sets; BCS gives real models.
        appearance_slots = {
            "FaceBase": (0, "Face_base"),
            "FaceForehead": (1, "Face_forehead"),
            "Eyes": (2, "Face_eye"),
            "Nose": (3, "Face_nose"),
            "Ears": (4, "Face_ear"),
            "Hair": (5, "Hair"),
        }
        for xml_name, (slot, suffix) in appearance_slots.items():
            assets.extend(_resolve_bcs_part_assets(
                chara_root, race, model_race, visual_bcs,
                profile.appearance.get(xml_name, -1), slot, suffix,
            ))

        preset = profile.active_preset()
        clothing_slots = {
            "Top": (6, "Bust"),
            "Bottom": (7, "Pants"),
            "Gloves": (8, "Rist"),
            "Shoes": (9, "Boots"),
        }
        unresolved_slots: list[str] = []
        for xml_name, (slot, suffix) in clothing_slots.items():
            selected_id = preset.equipment.get(xml_name, -1)
            resolved = _resolve_bcs_part_assets(
                chara_root, race, model_race, visual_bcs, selected_id, slot, suffix,
            )
            assets.extend(resolved)
            if selected_id >= 0 and not any(asset.kind == AssetKind.MODEL for asset in resolved):
                unresolved_slots.append(f"{xml_name}:{selected_id}")

        # Accessory lookup is intentionally retained as a conservative exact-ID
        # fallback because accessories are stored outside the ten standard BCS
        # visual slots in vanilla save data.
        accessory_id = preset.equipment.get("Accessory", -1)
        if accessory_id >= 0:
            exact_tokens = (f"_{accessory_id:03d}_", f"_{accessory_id + 1000:04d}_")
            for folder_code in dict.fromkeys((model_race, race)):
                folder = chara_root / folder_code
                if not folder.is_dir():
                    continue
                for path in folder.iterdir():
                    name = path.name.casefold()
                    if (
                        path.is_file()
                        and any(token.casefold() in name for token in exact_tokens)
                        and any(token in name for token in ("acc", "hair", "face"))
                        and "_lod" not in name
                    ):
                        assets.extend(_companions_for_stem(folder, path.stem, race))

        assets = _dedupe_assets(assets)
        # Prefer a race-specific rig, then the shared visual rig. Saiyans do
        # not necessarily ship SYM_000/SYF_000 in extracted layouts.
        skeleton_names = (f"{race}_000.esk", f"{model_race}_000.esk",
                          f"{race}.esk", f"{model_race}.esk")
        main_skeleton = next((
            a.path for name in skeleton_names for a in assets
            if a.kind == AssetKind.SKELETON
            and a.path.name.casefold() == name.casefold()
        ), None)
        if main_skeleton is not None:
            for asset in assets:
                if asset.kind == AssetKind.MODEL and asset.paired_skeleton is None and not asset.is_physics:
                    asset.paired_skeleton = main_skeleton

        main_animation = next((
            a.path for a in assets
            if a.kind == AssetKind.ANIMATION
            and a.path.name.casefold() == f"{race}.ean".casefold()
        ), None)
        code = f"CAC_{profile.slot:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', profile.name).strip('_') or 'Character'}"
        metadata: dict[str, Any] = {
            "cac": True, "cac_slot": profile.slot, "cac_profile": asdict(profile),
            "race": race, "race_name": profile.race_name,
            "model_race": model_race, "model_folder": str(model_folder),
            "bcs_file": str(bcs_path), "bcs_loaded": visual_bcs is not None,
            "unresolved_visual_slots": unresolved_slots,
            "preset_index": preset.index,
            "main_skeleton": str(main_skeleton) if main_skeleton else "",
            "main_animation": str(main_animation) if main_animation else "",
            "export_name": profile.name,
        }
        entries.append(LibraryEntry(
            path=race_folder, category="CaC", code=code, display_name=profile.name,
            usable=True, id_record=None, assets=assets, metadata=metadata,
        ))
    return entries

