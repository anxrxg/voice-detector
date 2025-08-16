"""Dataset organizer to populate the expected data/ layout.

Expected final layout (relative to --project-root):
  data/
    gender/
      male/   *.wav|*.flac|*.ogg|*.mp3
      female/ *.wav|*.flac|*.ogg|*.mp3
    age/
      wav/    *.wav|*.flac|*.ogg|*.mp3    + optional CSV labels at data/age/age_labels.csv
    emotion/
      wav/    *.wav|*.flac|*.ogg|*.mp3    + optional CSV labels at data/emotion/emotion_labels.csv

Usage examples:
  # Copy gender data from arbitrary folders into the project structure
  python organize_dataset.py --project-root . \
    --gender-src-male /path/to/male \
    --gender-src-female /path/to/female

  # Age and emotion with labels CSVs
  python organize_dataset.py --project-root . \
    --age-src /path/to/age_audio --age-labels /path/to/age_labels.csv \
    --emotion-src /path/to/emo_audio --emotion-labels /path/to/emotion_labels.csv

  # Use symlinks instead of copying to save space
  python organize_dataset.py --project-root . --link
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def _iter_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            yield p


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_or_link(src: Path, dst: Path, use_link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_link:
        try:
            # Prefer symlink; fall back to hardlink if symlink fails on platform
            try:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src)
            except Exception:
                if dst.exists():
                    dst.unlink()
                dst.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def _copy_tree_of_audio(src_root: Path, dst_root: Path, use_link: bool) -> int:
    count = 0
    for src in _iter_audio_files(src_root):
        rel_name = src.name
        dst = dst_root / rel_name
        _copy_or_link(src, dst, use_link)
        count += 1
    return count


def _copy_labels_csv(labels_src: Optional[Path], labels_dst: Path) -> Optional[Path]:
    if labels_src is None:
        return None
    labels_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_src, labels_dst)
    return labels_dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize dataset into project data/ structure")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).parent))

    # Sources (optional; specify what you have)
    parser.add_argument("--gender-src-male", type=str, default=None, help="Folder with male audio")
    parser.add_argument("--gender-src-female", type=str, default=None, help="Folder with female audio")
    parser.add_argument("--age-src", type=str, default=None, help="Folder with age audio")
    parser.add_argument("--age-labels", type=str, default=None, help="CSV with age labels (filename,label)")
    parser.add_argument("--emotion-src", type=str, default=None, help="Folder with emotion audio")
    parser.add_argument("--emotion-labels", type=str, default=None, help="CSV with emotion labels (filename,label)")

    parser.add_argument("--link", action="store_true", help="Symlink instead of copying files")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()

    # Targets
    gender_male_dst = root / "data/gender/male"
    gender_female_dst = root / "data/gender/female"
    age_dst = root / "data/age/wav"
    emotion_dst = root / "data/emotion/wav"

    # Ensure dirs exist
    for d in (gender_male_dst, gender_female_dst, age_dst, emotion_dst):
        _ensure_dir(d)

    total = 0

    # Gender
    if args.gender_src_male:
        n = _copy_tree_of_audio(Path(args.gender_src_male), gender_male_dst, args.link)
        print(f"Copied/linked {n} male files -> {gender_male_dst}")
        total += n
    if args.gender_src_female:
        n = _copy_tree_of_audio(Path(args.gender_src_female), gender_female_dst, args.link)
        print(f"Copied/linked {n} female files -> {gender_female_dst}")
        total += n

    # Age
    if args.age_src:
        n = _copy_tree_of_audio(Path(args.age_src), age_dst, args.link)
        print(f"Copied/linked {n} age files -> {age_dst}")
        total += n
    if args.age_labels:
        labels_dst = root / "data/age/age_labels.csv"
        out = _copy_labels_csv(Path(args.age_labels), labels_dst)
        if out:
            print(f"Copied labels -> {out}")

    # Emotion
    if args.emotion_src:
        n = _copy_tree_of_audio(Path(args.emotion_src), emotion_dst, args.link)
        print(f"Copied/linked {n} emotion files -> {emotion_dst}")
        total += n
    if args.emotion_labels:
        labels_dst = root / "data/emotion/emotion_labels.csv"
        out = _copy_labels_csv(Path(args.emotion_labels), labels_dst)
        if out:
            print(f"Copied labels -> {out}")

    if total == 0:
        print("No source folders provided or no audio found. Nothing to do.")
    else:
        print(f"Done. Total files placed: {total}")


if __name__ == "__main__":
    main()
