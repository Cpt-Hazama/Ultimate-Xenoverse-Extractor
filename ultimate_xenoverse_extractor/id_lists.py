from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from .models import IdRecord


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _truthy(value: str) -> bool | None:
    text = value.strip().casefold()
    if text in {"yes", "true", "y", "1", "playable", "usable"}:
        return True
    if text in {"no", "false", "n", "0", "unused", "unusable"}:
        return False
    return None


def parse_html_table(path: Path) -> tuple[list[str], list[list[str]]]:
    parser = _TableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    rows = [row for row in parser.rows if len(row) > 1]
    if not rows:
        return [], []

    # Google Sheets exports prepend a row containing A/B/C column labels. The
    # real header is the first row containing an ID field plus a useful field.
    header_index = next(
        (
            i
            for i, row in enumerate(rows)
            if any(any(token.startswith("id") for token in _normalize_header(cell).split()) for cell in row)
            and any(
                token in _normalize_header(" ".join(row)).split()
                for token in ("code", "name", "playable", "cac")
            )
        ),
        0,
    )
    return rows[header_index], rows[header_index + 1 :]


def _column_exact(headers: list[str], *aliases: str) -> int | None:
    normalized_aliases = {_normalize_header(alias) for alias in aliases}
    for index, header in enumerate(headers):
        if _normalize_header(header) in normalized_aliases:
            return index
    return None


def _column_contains(headers: list[str], *needles: str) -> int | None:
    normalized_needles = tuple(_normalize_header(needle) for needle in needles)
    for index, header in enumerate(headers):
        normalized = _normalize_header(header)
        if all(needle in normalized for needle in normalized_needles):
            return index
    return None


def parse_id_file(path: Path) -> list[IdRecord]:
    headers, rows = parse_html_table(path)
    if not headers:
        return []

    # Character/stage/etc. sheets use ID (NUM)/ID (HEX), whereas skill sheets
    # use ID1 (Long)/ID2 (Short). Store both forms in the existing IdRecord
    # fields so the rest of the application remains backwards compatible.
    primary_id_i = _column_exact(
        headers,
        "ID (NUM)",
        "ID NUM",
        "ID1 (Long)",
        "ID1 Long",
        "ID Long",
        "ID",
        "Model ID",
    )
    if primary_id_i is None:
        primary_id_i = _column_contains(headers, "id", "num")
    if primary_id_i is None:
        primary_id_i = _column_contains(headers, "id1")

    secondary_id_i = _column_exact(
        headers,
        "ID (HEX)",
        "ID HEX",
        "ID2 (Short)",
        "ID2 Short",
        "ID Short",
    )
    if secondary_id_i is None:
        secondary_id_i = _column_contains(headers, "id", "hex")
    if secondary_id_i is None:
        secondary_id_i = _column_contains(headers, "id2")

    code_i = _column_exact(headers, "Code")
    if code_i is None:
        code_i = _column_contains(headers, "code")

    name_i = _column_exact(headers, "Name")
    if name_i is None:
        name_i = _column_contains(headers, "name")

    usable_i = _column_contains(headers, "playable")
    if usable_i is None:
        usable_i = _column_contains(headers, "cac")

    notes_i = _column_contains(headers, "notes")

    def get(row: list[str], index: int | None) -> str:
        return row[index].strip() if index is not None and index < len(row) else ""

    records: list[IdRecord] = []
    for row in rows:
        primary_id = get(row, primary_id_i)
        secondary_id = get(row, secondary_id_i)
        code = get(row, code_i)
        name = get(row, name_i)

        # Skip the empty spacer row present in every Google Sheets export.
        if not any((primary_id, secondary_id, code, name)):
            continue

        records.append(
            IdRecord(
                numeric_id=primary_id,
                hex_id=secondary_id,
                code=code,
                name=name or code or primary_id or secondary_id,
                usable=_truthy(get(row, usable_i)) if usable_i is not None else None,
                notes=get(row, notes_i),
                source=path.stem,
            )
        )
    return records


def load_all_id_lists(directory: Path) -> dict[str, list[IdRecord]]:
    result: dict[str, list[IdRecord]] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("*.html")):
        result[path.stem] = parse_id_file(path)
    return result


def _id_aliases(value: str) -> set[str]:
    value = value.strip().upper()
    if not value:
        return set()

    aliases = {value}
    if value.isdigit():
        # Folder IDs are often not zero-padded while character IDs are.
        aliases.add(str(int(value)))
        aliases.add(value.zfill(3))
        aliases.add(value.zfill(4))
        aliases.add(value.zfill(5))
    return aliases


def index_records(records: Iterable[IdRecord]) -> dict[str, IdRecord]:
    index: dict[str, IdRecord] = {}
    for record in records:
        keys: set[str] = set()
        keys.update(_id_aliases(record.code))
        keys.update(_id_aliases(record.numeric_id))
        keys.update(_id_aliases(record.hex_id))
        for key in keys:
            index.setdefault(key, record)
    return index
