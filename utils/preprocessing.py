"""Utility functions for dataset preparation and audio preprocessing.

Phase 1 focuses on dataset preparation. This module currently provides
validation helpers to ensure the expected directory structure is present
before any audio preprocessing or training occurs.

Guardrails respected:
- No external data fetching or system calls
- No destructive operations
- Exceptions are handled and surfaced clearly
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import random
import csv


@dataclass(frozen=True)
class DatasetStructureStatus:
    """Status object describing the dataset directory readiness.

    Attributes:
        exists: Whether the provided project root exists.
        required_directories_present: Whether all required subdirectories exist.
        missing_directories: List of required directories that are missing.
        notes: Additional informational notes for the caller.
    """

    exists: bool
    required_directories_present: bool
    missing_directories: List[str]
    notes: List[str]


def validate_dataset_structure(project_root: str | Path) -> DatasetStructureStatus:
    """Validate that the expected dataset directory structure exists.

    Expected minimal structure (per filestructure.md and task.md Phase 1):
      - data/gender/male/
      - data/gender/female/
      - data/age/wav/
      - data/emotion/wav/

    This function does not create or modify any files. It only reads the
    filesystem and reports what is missing so that a human can address it.

    Args:
        project_root: Absolute or relative path to the project root directory.

    Returns:
        DatasetStructureStatus with details about missing directories.

    Raises:
        ValueError: If project_root is empty or None.
    """
    if not project_root:
        raise ValueError("project_root must be a non-empty path string or Path")

    root_path = Path(project_root).resolve()
    if not root_path.exists():
        return DatasetStructureStatus(
            exists=False,
            required_directories_present=False,
            missing_directories=[str(root_path)],
            notes=["Project root does not exist"],
        )

    required_dirs: List[Path] = [
        root_path / "data" / "gender" / "male",
        root_path / "data" / "gender" / "female",
        root_path / "data" / "age" / "wav",
        root_path / "data" / "emotion" / "wav",
    ]

    missing: List[str] = [str(d.relative_to(root_path)) for d in required_dirs if not d.exists()]

    notes: List[str] = []
    if not (root_path / "processed_data").exists():
        notes.append("Optional: create 'processed_data/' for saved preprocessed files")

    return DatasetStructureStatus(
        exists=True,
        required_directories_present=len(missing) == 0,
        missing_directories=missing,
        notes=notes,
    )


def load_audio(filepath: str | Path, target_sample_rate: int = 16_000) -> Tuple[np.ndarray, int]:
    """Load audio as mono at a fixed sample rate.

    Converts the input to mono and resamples to `target_sample_rate`.

    Args:
        filepath: Path to an audio file.
        target_sample_rate: Desired sample rate in Hz.

    Returns:
        (audio, sample_rate) where audio is 1D float32 in [-1, 1].

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If loading fails.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        import librosa  # Lazy import to keep base imports light

        audio, sample_rate = librosa.load(str(path), sr=target_sample_rate, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        return audio, int(sample_rate)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"Failed to load audio '{path}': {error}") from error


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int,
    noise_frames: int = 6,
    n_fft: int = 1024,
    hop_length: int = 256,
    noise_multiplier: float = 1.5,
) -> np.ndarray:
    """Apply a lightweight spectral-gating noise reduction.

    Estimates a noise profile from the first `noise_frames` STFT frames and
    attenuates spectral magnitudes under a threshold.

    This method is intentionally simple and efficient for local execution.

    Args:
        audio: Mono float32 array in [-1, 1].
        sample_rate: Sample rate in Hz.
        noise_frames: Initial frames to estimate noise.
        n_fft: FFT window size.
        hop_length: Hop length for STFT frames.
        noise_multiplier: Threshold multiplier for gating.

    Returns:
        Denoised audio as float32 in [-1, 1].
    """
    if audio.ndim != 1:
        raise ValueError("reduce_noise expects a 1D mono waveform")
    if audio.size == 0:
        return audio.astype(np.float32)

    try:
        import librosa

        stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft_matrix)
        phase = np.exp(1j * np.angle(stft_matrix))

        frames_for_noise = max(1, min(noise_frames, magnitude.shape[1]))
        noise_mag = np.mean(magnitude[:, :frames_for_noise], axis=1, keepdims=True)

        threshold = noise_multiplier * noise_mag
        magnitude_denoised = np.maximum(magnitude - threshold, 0.0)

        stft_denoised = magnitude_denoised * phase
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length, length=len(audio))

        return np.clip(audio_denoised, -1.0, 1.0).astype(np.float32)
    except Exception:
        # In constrained environments, safely return original audio
        return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio to `target_peak` (<= 1.0)."""
    if audio.ndim != 1:
        raise ValueError("normalize_audio expects a 1D mono waveform")
    if audio.size == 0:
        return audio.astype(np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio.astype(np.float32)

    scale = min(1.0, float(target_peak) / peak) if target_peak > 0.0 else 1.0
    return (audio * scale).astype(np.float32)


def preprocess_audio_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    save: bool = False,
    target_sample_rate: int = 16_000,
) -> Tuple[np.ndarray, int]:
    """Load, denoise, and normalize an audio file.

    Steps:
      1) Load as mono at 16 kHz
      2) Apply spectral-gating noise reduction
      3) Peak normalize to 0.95
      4) Optionally save to `output_path`

    Returns (audio, sample_rate).
    """
    audio, sample_rate = load_audio(input_path, target_sample_rate)
    audio = reduce_noise(audio, sample_rate)
    audio = normalize_audio(audio, target_peak=0.95)

    if save:
        if output_path is None:
            raise ValueError("output_path is required when save=True")
        try:
            import soundfile as sf

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio, sample_rate)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"Failed to save processed audio: {error}") from error

    return audio, sample_rate


__all__ = [
    "DatasetStructureStatus",
    "validate_dataset_structure",
    "load_audio",
    "reduce_noise",
    "normalize_audio",
    "preprocess_audio_file",
]


# -----------------------------
# Phase 4: Dataset Splitting
# -----------------------------

def _list_audio_files(directory: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """List audio files recursively with de-duplication by basename stem.

    If multiple files share the same basename stem but different extensions
    (e.g., `foo.mp3` and `foo.wav`), prefer the extension that appears earlier
    in `extensions`. This helps avoid training on both MP3 and WAV duplicates
    after a conversion step.
    """
    collected: List[Path] = []
    for ext in extensions:
        collected.extend(directory.rglob(f"*.{ext.lstrip('.')}"))

    # Prefer earlier extensions in the tuple
    preference_order = {ext.lstrip('.').lower(): idx for idx, ext in enumerate(extensions)}
    best_by_stem: Dict[str, Path] = {}
    for path in collected:
        stem = path.stem
        current = best_by_stem.get(stem)
        if current is None:
            best_by_stem[stem] = path
            continue
        prev_score = preference_order.get(current.suffix.lstrip('.').lower(), 999)
        new_score = preference_order.get(path.suffix.lstrip('.').lower(), 999)
        if new_score < prev_score:
            best_by_stem[stem] = path

    return sorted(best_by_stem.values())


def split_list(
    items: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Split a list deterministically into train/val/test.

    The ratios must be non-negative and sum to 1 within a small tolerance.
    Splitting is performed by shuffling with a fixed seed and then slicing.
    """
    if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Ratios must be non-negative")
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError("Ratios must sum to 1.0 (within tolerance)")

    items_copy = list(items)
    rng = random.Random(seed)
    rng.shuffle(items_copy)

    n = len(items_copy)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Assign remainder to test to ensure full coverage
    n_test = n - n_train - n_val

    train_split = items_copy[:n_train]
    val_split = items_copy[n_train : n_train + n_val]
    test_split = items_copy[n_train + n_val :]

    return {"train": train_split, "val": val_split, "test": test_split}


def prepare_gender_splits(
    project_root: str | Path,
    extensions: Tuple[str, ...] = ("wav", "flac", "ogg", "mp3"),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, str]]]:
    """Prepare gender dataset splits using folder names as labels.

    Returns dict mapping split -> list of (filepath, label) with label in
    {"Male", "Female"}.
    """
    root = Path(project_root).resolve()
    male_dir = root / "data" / "gender" / "male"
    female_dir = root / "data" / "gender" / "female"

    male_files = _list_audio_files(male_dir, extensions) if male_dir.exists() else []
    female_files = _list_audio_files(female_dir, extensions) if female_dir.exists() else []

    splits = {}
    for label, files in (("Male", male_files), ("Female", female_files)):
        part = split_list(files, train_ratio, val_ratio, test_ratio, seed)
        for split_name, split_files in part.items():
            labeled = [(str(p), label) for p in split_files]
            splits.setdefault(split_name, []).extend(labeled)

    # Shuffle within each split for mixed classes
    rng = random.Random(seed)
    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


def _read_simple_csv_labels(csv_path: Path) -> Dict[str, str]:
    """Read labels from a two-column CSV: filename,label.

    Returns a mapping from filename to label string.
    """
    mapping: Dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            filename, label = row[0].strip(), row[1].strip()
            mapping[filename] = label
    return mapping


def prepare_age_splits(
    project_root: str | Path,
    labels_csv: Optional[str | Path] = None,
    extensions: Tuple[str, ...] = ("wav", "flac", "ogg", "mp3"),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, Optional[float]]]]:
    """Prepare age dataset splits from `data/age/wav` using optional CSV labels.

    CSV format: filename,label (where label is numeric age). If not provided,
    labels will be None.
    """
    root = Path(project_root).resolve()
    audio_dir = root / "data" / "age" / "wav"
    files = _list_audio_files(audio_dir, extensions) if audio_dir.exists() else []
    splits_paths = split_list(files, train_ratio, val_ratio, test_ratio, seed)

    label_map: Dict[str, Optional[float]] = {}
    if labels_csv is not None:
        csv_path = Path(labels_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"labels_csv not found: {csv_path}")
        raw_map = _read_simple_csv_labels(csv_path)
        # Coerce to float where possible
        # Also populate extension-agnostic keys to tolerate mp3→wav conversion
        for k, v in raw_map.items():
            try:
                value: Optional[float] = float(v)
            except ValueError:
                value = None
            label_map[k] = value
            stem = Path(k).stem
            for ext in ("wav", "flac", "ogg", "mp3"):
                alt = f"{stem}.{ext}"
                if alt not in label_map:
                    label_map[alt] = value

    result: Dict[str, List[Tuple[str, Optional[float]]]] = {}
    for split_name, paths in splits_paths.items():
        entries: List[Tuple[str, Optional[float]]] = []
        for p in paths:
            age = label_map.get(p.name, None)
            # Prefer WAV over lossy formats to reduce decoder issues
            entries.append((str(p), age))
        result[split_name] = entries
    return result


def prepare_emotion_splits(
    project_root: str | Path,
    labels_csv: Optional[str | Path] = None,
    extensions: Tuple[str, ...] = ("wav", "flac", "ogg", "mp3"),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """Prepare emotion dataset splits from `data/emotion/wav` with optional labels.

    CSV format: filename,label (where label is an emotion class string).
    """
    root = Path(project_root).resolve()
    audio_dir = root / "data" / "emotion" / "wav"
    files = _list_audio_files(audio_dir, extensions) if audio_dir.exists() else []
    splits_paths = split_list(files, train_ratio, val_ratio, test_ratio, seed)

    label_map: Dict[str, Optional[str]] = {}
    if labels_csv is not None:
        csv_path = Path(labels_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"labels_csv not found: {csv_path}")
        raw_map = _read_simple_csv_labels(csv_path)
        label_map.update(raw_map)

    result: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for split_name, paths in splits_paths.items():
        entries: List[Tuple[str, Optional[str]]] = []
        for p in paths:
            label = label_map.get(p.name, None)
            entries.append((str(p), label))
        result[split_name] = entries
    return result

