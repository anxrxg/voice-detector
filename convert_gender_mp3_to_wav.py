"""Convert Common Voice gender MP3 files to WAV to avoid decode issues.

- Scans data/gender/{male,female} for *.mp3
- Writes corresponding *.wav next to each MP3 (same basename)
- Requires ffmpeg in PATH

Usage:
  python convert_gender_mp3_to_wav.py --project-root . --workers 4

Then train using WAVs only to bypass MP3 decode:
  python train.py --task gender --num-workers 4

"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _convert_one(src: Path, dst: Path) -> tuple[Path, bool, str]:
    if dst.exists():
        return (dst, True, "exists")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return (dst, True, "ok")
    except Exception as e:  # noqa: BLE001
        return (dst, False, str(e))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert gender MP3s to WAV")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).parent))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    if not _has_ffmpeg():
        raise SystemExit("ffmpeg not found. Install it (e.g., brew install ffmpeg) and retry.")

    mp3_files: List[Path] = []
    for sub in ("male", "female"):
        d = root / "data" / "gender" / sub
        if d.exists():
            mp3_files.extend(sorted(d.glob("*.mp3")))

    print(f"Found {len(mp3_files)} MP3 files to convert")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = []
        for mp3 in mp3_files:
            wav = mp3.with_suffix(".wav")
            futures.append(ex.submit(_convert_one, mp3, wav))
        ok = 0
        fail = 0
        for fut in as_completed(futures):
            dst, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(f"[WARN] Failed: {dst.name}: {msg}")

    print(f"Done. Success: {ok}, Failures: {fail}")


if __name__ == "__main__":
    main()
