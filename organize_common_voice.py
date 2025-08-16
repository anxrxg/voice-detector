"""Organize a Mozilla Common Voice dump into this project's data/ layout.

- Source expected: a folder with subfolders like cv-valid-train/, cv-valid-dev/, etc.,
  and CSVs like cv-valid-train.csv with columns including filename, age, gender.
- We ignore cv-invalid.* by default.
- We place files as symlinks (or copies) under:
    data/gender/{male,female}/
    data/age/wav/
  and create numeric age labels at data/age/age_labels.csv (midpoint mapping).

Usage:
  python organize_common_voice.py \
    --project-root . \
    --source /absolute/path/to/voice-dataset \
    --link  # use symlinks instead of copying

Notes:
- We rename destination files to avoid collisions by encoding the split path into
  the filename: e.g., cv-valid-train__sample-000001.mp3
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}
AGE_MAP: Dict[str, float] = {
    "teens": 15.0,
    "twenties": 25.0,
    "thirties": 35.0,
    "forties": 45.0,
    "fifties": 55.0,
    "sixties": 65.0,
    "seventies": 75.0,
    "eighties": 85.0,
    "nineties": 95.0,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_or_link(src: Path, dst: Path, use_link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_link:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def _dest_basename(rel_path: str) -> str:
    # Encode subdir into filename to avoid collisions
    rel_path = rel_path.strip().lstrip("/\\")
    return rel_path.replace("/", "__").replace("\\", "__")


def _iter_csv_rows(source_root: Path):
    for csv_path in sorted(source_root.glob("cv-*.csv")):
        if csv_path.name.startswith("cv-invalid"):
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize Common Voice dataset into project data/")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--source", type=str, required=True, help="Path to Common Voice folder")
    parser.add_argument("--link", action="store_true", help="Symlink files instead of copying")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    source_root = Path(args.source).resolve()

    if not source_root.exists():
        raise SystemExit(f"Source not found: {source_root}")

    # Targets
    gender_male_dir = project_root / "data/gender/male"
    gender_female_dir = project_root / "data/gender/female"
    age_audio_dir = project_root / "data/age/wav"
    age_labels_csv = project_root / "data/age/age_labels.csv"

    _ensure_dir(gender_male_dir)
    _ensure_dir(gender_female_dir)
    _ensure_dir(age_audio_dir)
    _ensure_dir(age_labels_csv.parent)

    # Prepare age labels file (append-safe)
    age_labels_fp = age_labels_csv.open("w", newline="", encoding="utf-8")
    age_writer = csv.writer(age_labels_fp)
    age_writer.writerow(["filename", "label"])  # header

    n_processed = 0
    n_gender_male = 0
    n_gender_female = 0
    n_age = 0

    for row in _iter_csv_rows(source_root):
        rel = (row.get("filename") or "").strip()
        if not rel:
            continue
        src = source_root / rel
        if not src.suffix.lower() in AUDIO_EXTENSIONS:
            continue
        if not src.exists():
            # Handle datasets where audio is nested one extra level, e.g.:
            #   CSV:  cv-valid-train/sample-000001.mp3
            #   Path: cv-valid-train/cv-valid-train/sample-000001.mp3
            parts = rel.split("/", 1)
            if len(parts) == 2:
                top = parts[0]
                candidate = source_root / top / rel
                if candidate.exists():
                    src = candidate
                else:
                    # If still not found, skip safely (could be a stale CSV row)
                    continue
            else:
                # If still not found, skip safely (could be a stale CSV row)
                continue

        dst_base = _dest_basename(rel)

        # Gender placement
        gender = (row.get("gender") or "").strip().lower()
        if gender == "male":
            _copy_or_link(src, gender_male_dir / dst_base, args.link)
            n_gender_male += 1
        elif gender == "female":
            _copy_or_link(src, gender_female_dir / dst_base, args.link)
            n_gender_female += 1

        # Age placement + labels
        age_text = (row.get("age") or "").strip().lower()
        if age_text in AGE_MAP:
            age_value = AGE_MAP[age_text]
            _copy_or_link(src, age_audio_dir / dst_base, args.link)
            age_writer.writerow([dst_base, f"{age_value}"])
            n_age += 1

        n_processed += 1
        if args.limit is not None and n_processed >= int(args.limit):
            break

    age_labels_fp.close()

    print("Done organizing Common Voice → data/ structure")
    print(f"Processed rows:      {n_processed}")
    print(f"Gender male files:   {n_gender_male}")
    print(f"Gender female files: {n_gender_female}")
    print(f"Age labeled files:   {n_age}")
    print(f"Age labels CSV:      {age_labels_csv if n_age>0 else 'none'}")


if __name__ == "__main__":
    main()
