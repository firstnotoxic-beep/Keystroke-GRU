"""Subject registry helpers for keystroke data collection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict

from config import RAW_DATA_DIR, SUBJECTS_REGISTRY_PATH, normalize_subject_name

_INVALID_FILENAME_CHARS = r'\/:*?"<>|'

DEFAULT_SUBJECTS: list[dict[str, str | int]] = [{"name": "Owner", "label": 1}]


class SubjectEntry(TypedDict):
    name: str
    label: int


def validate_subject_name(name: str) -> tuple[bool, str]:
    """Validate a new subject name; return (ok, cleaned_name_or_error_message)."""
    stripped = name.strip()
    if not stripped:
        return False, "กรุณากรอกชื่อ Subject"
    for ch in _INVALID_FILENAME_CHARS:
        if ch in stripped:
            return False, f"ชื่อมีอักขระต้องห้าม: {ch}"
    if not normalize_subject_name(stripped):
        return False, "ชื่อ Subject ไม่ถูกต้อง"
    return True, stripped


def load_subjects_registry(path: Path | None = None) -> list[SubjectEntry]:
    """Load subjects from JSON; return default [Owner] if file is missing or invalid."""
    registry_path = path or SUBJECTS_REGISTRY_PATH
    if not registry_path.exists():
        return [SubjectEntry(name=str(s["name"]), label=int(s["label"])) for s in DEFAULT_SUBJECTS]

    try:
        with registry_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list) or not data:
            return [SubjectEntry(name=str(s["name"]), label=int(s["label"])) for s in DEFAULT_SUBJECTS]

        result: list[SubjectEntry] = []
        for item in data:
            if not isinstance(item, dict) or "name" not in item or "label" not in item:
                continue
            result.append(SubjectEntry(name=str(item["name"]).strip(), label=int(item["label"])))

        if not result:
            return [SubjectEntry(name=str(s["name"]), label=int(s["label"])) for s in DEFAULT_SUBJECTS]
        return result
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return [SubjectEntry(name=str(s["name"]), label=int(s["label"])) for s in DEFAULT_SUBJECTS]


def save_subjects_registry(
    subjects: list[SubjectEntry],
    path: Path | None = None,
) -> None:
    """Persist subject list to JSON."""
    registry_path = path or SUBJECTS_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as fh:
        json.dump(subjects, fh, ensure_ascii=False, indent=2)


def discover_subjects_from_raw(raw_dir: Path) -> list[SubjectEntry]:
    """Scan raw_data_*.csv and extract Subject_Name + Label from the first data row."""
    discovered: list[SubjectEntry] = []
    if not raw_dir.is_dir():
        return discovered

    seen: set[str] = set()
    for path in sorted(raw_dir.glob("raw_data_*.csv")):
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = (row.get("Subject_Name") or "").strip()
                    if not name:
                        continue
                    key = normalize_subject_name(name)
                    if key in seen:
                        break
                    try:
                        label = int(row["Label"])
                    except (KeyError, TypeError, ValueError):
                        break
                    discovered.append(SubjectEntry(name=name, label=label))
                    seen.add(key)
                    break
        except (OSError, csv.Error):
            continue
    return discovered


def merge_subject_lists(
    registry: list[SubjectEntry],
    discovered: list[SubjectEntry],
) -> list[SubjectEntry]:
    """Merge registry and discovered subjects; registry wins on label conflicts."""
    merged: dict[str, SubjectEntry] = {}

    for item in discovered:
        key = normalize_subject_name(item["name"])
        if key:
            merged[key] = SubjectEntry(name=item["name"], label=int(item["label"]))

    for item in registry:
        key = normalize_subject_name(item["name"])
        if key:
            merged[key] = SubjectEntry(name=item["name"], label=int(item["label"]))

    subjects = list(merged.values())
    subjects.sort(key=lambda s: (0 if s["label"] == 1 else 1, s["name"].lower()))
    return subjects


def get_all_subjects(raw_dir: Path | None = None) -> list[SubjectEntry]:
    """Return merged subject list from registry and raw CSV files."""
    raw_dir = raw_dir or RAW_DATA_DIR
    registry = load_subjects_registry()
    discovered = discover_subjects_from_raw(raw_dir)
    return merge_subject_lists(registry, discovered)


def find_subject_by_key(subjects: list[SubjectEntry], key: str) -> SubjectEntry | None:
    """Find subject entry by normalized name."""
    normalized = normalize_subject_name(key)
    for item in subjects:
        if normalize_subject_name(item["name"]) == normalized:
            return item
    return None


def is_duplicate_subject(name: str, subjects: list[SubjectEntry]) -> bool:
    """True if normalized name already exists in subject list."""
    key = normalize_subject_name(name)
    return any(normalize_subject_name(s["name"]) == key for s in subjects)
