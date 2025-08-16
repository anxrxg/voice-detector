"""Organize RAVDESS and CREMA-D into data/emotion/wav + labels CSV.

- Sources:
  --ravdess /path/to/RAVDESS
  --cremad  /path/to/CREMA-D

- Output under <project_root>:
  data/emotion/wav/*.wav   (symlink or copy)
  data/emotion/emotion_labels.csv  (filename,label)

- Labels inferred from filenames:
  RAVDESS: 03-01-EMO-...-*.wav with EMO code mapping:
    01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised
  CREMA-D: *_EMO_*.wav with EMO in {ANG, DIS, FEA, HAP, NEU, SAD} mapping to
    angry, disgust, fear, happy, neutral, sad

Usage:
  python organize_emotion_datasets.py --project-root . \
    --ravdess "/abs/path/RAVDESS" --cremad "/abs/path/CREMA-D" --link
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
RAVDESS_EMO = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised",
}
CREMAD_EMO = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
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


def _dest_basename(prefix: str, rel_path: Path) -> str:
    parts = [prefix] + [p for p in rel_path.parts]
    return "__".join(parts)


def _iter_ravdess(root: Path) -> Iterable[Tuple[Path, str]]:
    for p in root.rglob("*.wav"):
        name = p.name
        # Expect pattern like 03-01-05-01-02-01-12.wav
        fields = name.split("-")
        if len(fields) < 3:
            continue
        emo_code = fields[2]
        emo = RAVDESS_EMO.get(emo_code)
        if not emo:
            continue
        yield p, emo


def _iter_cremad(root: Path) -> Iterable[Tuple[Path, str]]:
    for p in root.rglob("*.wav"):
        name = p.stem  # e.g., 1001_DFA_ANG_XX
        parts = name.split("_")
        if len(parts) < 3:
            continue
        emo_code = parts[2].upper()
        emo = CREMAD_EMO.get(emo_code)
        if not emo:
            continue
        yield p, emo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize emotion datasets into project data/")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--ravdess", type=str, default=None, help="Path to RAVDESS root (Actor_* folders)")
    parser.add_argument("--cremad", type=str, default=None, help="Path to CREMA-D root (wav files)")
    parser.add_argument("--link", action="store_true", help="Symlink instead of copy")
    parser.add_argument("--limit", type=int, default=None, help="Optional max files to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "data/emotion/wav"
    labels_csv = project_root / "data/emotion/emotion_labels.csv"
    _ensure_dir(out_dir)
    _ensure_dir(labels_csv.parent)

    # Open labels CSV for writing fresh
    f = labels_csv.open("w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])  # header

    count = 0

    if args.ravdess:
        root = Path(args.ravdess)
        for src, emo in _iter_ravdess(root):
            rel = src.relative_to(root)
            dst_name = _dest_basename("RAVDESS", rel)
            _copy_or_link(src, out_dir / dst_name, args.link)
            writer.writerow([dst_name, emo])
            count += 1
            if args.limit is not None and count >= args.limit:
                break

    if (args.limit is None or count < args.limit) and args.cremad:
        root = Path(args.cremad)
        for src, emo in _iter_cremad(root):
            rel = src.relative_to(root)
            dst_name = _dest_basename("CREMA-D", rel)
            _copy_or_link(src, out_dir / dst_name, args.link)
            writer.writerow([dst_name, emo])
            count += 1
            if args.limit is not None and count >= args.limit:
                break

    f.close()

    print("Done organizing emotion datasets → data/emotion/")
    print("Total files:", count)
    print("Labels CSV:", labels_csv if count else "none")


if __name__ == "__main__":
    main()
