"""Deterministic case discovery and source cataloguing with provenance.

A case is any folder containing `00_START_HERE.txt`. Nothing here branches on a specific
customer name. Original files are only read, never modified.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from .models import SourceRecord, canonical_lines

CASE_MARKER = "00_START_HERE.txt"
KIND_BY_SUFFIX = {".txt": "text", ".md": "markdown", ".csv": "csv"}


def case_id_for(folder_name: str) -> str:
    return folder_name.strip().lower().replace(" ", "_")


def canonical_text(raw: bytes) -> str:
    """Decode UTF-8 (BOM tolerated) and normalise CRLF/CR to LF. Line anchors use this."""
    text = raw.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def make_source_id(case_id: str, relative_path: str, sha256: str) -> str:
    # Case and path are part of the identity, so identical bytes in two cases never collide.
    return f"{case_id}:{relative_path}@{sha256[:12]}"


def discover_cases(root: Path) -> list[dict]:
    cases = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith(".")):
        marker = folder / CASE_MARKER
        if not marker.exists():
            continue
        first_line = canonical_lines(canonical_text(marker.read_bytes()))[:1]
        cases.append(
            {
                "case_id": case_id_for(folder.name),
                "folder": folder.name,
                "path": str(folder),
                "title": first_line[0] if first_line else folder.name,
            }
        )
    return cases


def scan_case(root: Path, case_id: str) -> tuple[list[SourceRecord], list[str]]:
    """Return source records and visible warnings for skipped files."""
    matches = [c for c in discover_cases(root) if c["case_id"] == case_id]
    if not matches:
        raise ValueError(f"Unknown case_id {case_id!r} under {root}")
    folder = Path(matches[0]["path"])
    records: list[SourceRecord] = []
    warnings: list[str] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        kind = KIND_BY_SUFFIX.get(path.suffix.lower())
        if kind is None:
            warnings.append(f"Skipped unsupported file type: {rel}")
            continue
        raw = path.read_bytes()
        try:
            text = canonical_text(raw)
        except UnicodeDecodeError:
            warnings.append(f"Skipped non-UTF-8 file: {rel}")
            continue
        sha = hashlib.sha256(raw).hexdigest()
        metadata = {
            "origin": "case_folder",
            "bytes": len(raw),
            "line_count": len(canonical_lines(text)),
            "trailing_newline": text.endswith("\n"),
        }
        if kind == "csv":
            metadata["rows"] = len(list(csv.DictReader(io.StringIO(text))))
        records.append(
            SourceRecord(
                source_id=make_source_id(case_id, rel, sha),
                case_id=case_id,
                relative_path=rel,
                content_sha256=sha,
                text=text,
                kind=kind,
                metadata=metadata,
            )
        )
    return records, warnings


def import_case(root: Path, case_id: str) -> list[SourceRecord]:
    return scan_case(root, case_id)[0]


def questionnaire_rows(sources: list[SourceRecord]) -> list[dict]:
    """Raw questionnaire rows (original values preserved) with their source line anchor."""
    rows: list[dict] = []
    for src in sources:
        if src.kind != "csv" or "questionnaire" not in src.relative_path:
            continue
        reader = csv.DictReader(io.StringIO(src.text))
        for row in reader:
            row = dict(row)
            # DictReader line_num is the physical line just read; questions here are single-line rows.
            row["_source_id"] = src.source_id
            row["_line"] = reader.line_num
            rows.append(row)
    return rows
