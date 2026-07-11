from __future__ import annotations

from collections import Counter
from pathlib import Path

from .id_lists import index_records
from .models import AssetFile, AssetKind, IdRecord, LibraryEntry

_EXT_KIND = {
    ".emd": AssetKind.MODEL,
    ".esk": AssetKind.SKELETON,
    ".emm": AssetKind.MATERIAL,
    ".emb": AssetKind.TEXTURE,
    ".ean": AssetKind.ANIMATION,
    ".bcs": AssetKind.CONFIG,
    ".bac": AssetKind.GAMEPLAY,
    ".bcm": AssetKind.GAMEPLAY,
    ".bdm": AssetKind.GAMEPLAY,
    ".bai": AssetKind.GAMEPLAY,
}

_CATEGORY_DIRS = {
    "chara": "Characters",
    "skill": "Skills",
    "stage": "Stages",
    "item": "Items",
}

# data/skill subdirectory -> matching HTML ID list.
_SKILL_LIST_BY_TYPE = {
    "SPA": "Super IDs",
    "ULT": "Ultimate IDs",
    "ESC": "Evasive IDs",
    "MET": "Awoken IDs",
    "BLT": "Ki Blast IDs",
}


def classify(path: Path) -> AssetKind:
    name = path.name.lower()
    if name.endswith(".cam.ean"):
        return AssetKind.CAMERA
    if name.endswith(".fce.lip.ean"):
        return AssetKind.LIP_ANIMATION
    if name.endswith(".fce.ean"):
        return AssetKind.FACE_ANIMATION
    return _EXT_KIND.get(path.suffix.lower(), AssetKind.OTHER)


def logical_name(path: Path) -> str:
    name = path.name
    for suffix in (".dyt.emb", ".cam.ean", ".fce.lip.ean", ".fce.ean"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def root_code_for_folder(folder: Path) -> str:
    return folder.name.upper()


def scan_assets(folder: Path) -> list[AssetFile]:
    code = root_code_for_folder(folder)
    files = [path for path in folder.rglob("*") if path.is_file()]
    skeletons = {path.stem.casefold(): path for path in files if path.suffix.lower() == ".esk"}
    assets: list[AssetFile] = []
    for path in sorted(files, key=lambda item: (item.name.casefold(), str(item).casefold())):
        kind = classify(path)
        stem = path.stem
        paired = skeletons.get(stem.casefold()) if kind == AssetKind.MODEL else None
        physics = kind == AssetKind.MODEL and paired is not None and stem.casefold() != f"{code}_000".casefold()
        assets.append(AssetFile(path, kind, logical_name(path), code, physics, paired))
    return assets


def _category_roots(base: Path) -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    for key, label in _CATEGORY_DIRS.items():
        candidate = base / key
        if candidate.is_dir():
            roots.append((candidate, label))
    return roots


def find_library_roots(selected: Path) -> list[tuple[Path, str]]:
    selected = selected.resolve()

    selected_key = selected.name.casefold()
    if selected_key in _CATEGORY_DIRS:
        return [(selected, _CATEGORY_DIRS[selected_key])]

    roots = _category_roots(selected)
    if roots:
        return roots

    data_dir = selected / "data"
    if data_dir.is_dir():
        roots = _category_roots(data_dir)
        if roots:
            return roots

    return [(selected, selected.name or "Library")]


def _records_named(all_lists: dict[str, list[IdRecord]], name: str) -> list[IdRecord]:
    for list_name, records in all_lists.items():
        if list_name.casefold() == name.casefold():
            return records
    return []


def _records_for_category(all_lists: dict[str, list[IdRecord]], category: str) -> list[IdRecord]:
    preferred = {
        "characters": "Character IDs",
        "stages": "Stage IDs",
        "items": "Clothing IDs",
    }.get(category.casefold())
    if preferred:
        records = _records_named(all_lists, preferred)
        if records:
            return records
    return [record for records in all_lists.values() for record in records]


def _skill_records(all_lists: dict[str, list[IdRecord]], skill_type: str) -> list[IdRecord]:
    preferred_name = _SKILL_LIST_BY_TYPE.get(skill_type.upper())
    if preferred_name:
        records = _records_named(all_lists, preferred_name)
        if records:
            return records

    # CMN and unknown/custom skill categories can contain shared or modded data.
    # Search every skill-oriented list as a safe fallback.
    records: list[IdRecord] = []
    for list_name in (
        "Super IDs",
        "Ultimate IDs",
        "Evasive IDs",
        "Awoken IDs",
        "Ki Blast IDs",
    ):
        records.extend(_records_named(all_lists, list_name))
    return records


def _skill_lookup_keys(folder_name: str) -> list[str]:
    code = folder_name.upper()
    pieces = [piece for piece in code.split("_") if piece]
    keys = [code]

    # Examples:
    #   1091_CMN_MGA -> 1091, MGA, CMN_MGA
    #   000_GOK_SSJ  -> 000, GOK_SSJ, SSJ
    if pieces:
        keys.append(pieces[0])
    if len(pieces) > 1:
        keys.append("_".join(pieces[1:]))
        keys.append(pieces[-1])
    return list(dict.fromkeys(keys))


def _find_record(lookup: dict[str, IdRecord], keys: list[str]) -> IdRecord | None:
    for key in keys:
        normalized = key.strip().upper()
        if not normalized:
            continue
        record = lookup.get(normalized)
        if record is not None:
            return record
        if normalized.isdigit():
            record = lookup.get(str(int(normalized)))
            if record is not None:
                return record
    return None


def build_library(selected: Path, all_lists: dict[str, list[IdRecord]]) -> list[LibraryEntry]:
    entries: list[LibraryEntry] = []
    for root, category in find_library_roots(selected):
        if category == "Skills":
            folders = sorted(
                (
                    folder
                    for folder in root.rglob("*")
                    if folder.is_dir() and any(child.is_file() for child in folder.iterdir())
                ),
                key=lambda path: str(path.relative_to(root)).casefold(),
            )
        else:
            folders = sorted(
                (path for path in root.iterdir() if path.is_dir()),
                key=lambda path: path.name.casefold(),
            )

        category_lookup = index_records(_records_for_category(all_lists, category))

        for folder in folders:
            code = folder.name.upper()
            entry_category = category

            if category == "Skills":
                relative = folder.relative_to(root)
                skill_type = relative.parts[0].upper() if len(relative.parts) > 1 else "OTHER"
                entry_category = f"Skills / {skill_type}"
                lookup = index_records(_skill_records(all_lists, skill_type))
                record = _find_record(lookup, _skill_lookup_keys(code))
            else:
                record = _find_record(category_lookup, [code])

            display = record.name if record else code
            entries.append(
                LibraryEntry(
                    folder,
                    entry_category,
                    code,
                    display,
                    record.usable if record else None,
                    record,
                    scan_assets(folder),
                )
            )
    return entries


def summarize_assets(assets: list[AssetFile]) -> str:
    counts = Counter(asset.kind.value for asset in assets)
    return ", ".join(f"{value}: {count}" for value, count in sorted(counts.items())) or "No files"
