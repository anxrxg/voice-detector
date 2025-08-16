"""Feature extraction utilities for audio signals.

Provides MFCCs, Chroma, and Mel-spectrogram features with consistent shapes
and types, suitable for lightweight local processing.

All functions include error handling per project guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim != 1:
        raise ValueError("Expected 1D mono waveform for feature extraction")
    return np.asarray(audio, dtype=np.float32)


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Extract MFCC features.

    Returns an array of shape (n_mfcc, time_frames) as float32.
    """
    audio = _ensure_mono_float32(audio)
    if audio.size == 0:
        return np.zeros((n_mfcc, 0), dtype=np.float32)
    try:
        import librosa

        mfcc = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        return np.asarray(mfcc, dtype=np.float32)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"MFCC extraction failed: {error}") from error


def extract_chroma(
    audio: np.ndarray,
    sample_rate: int,
    n_chroma: int = 12,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Extract Chroma features.

    Returns array of shape (n_chroma, time_frames) as float32.
    """
    audio = _ensure_mono_float32(audio)
    if audio.size == 0:
        return np.zeros((n_chroma, 0), dtype=np.float32)
    try:
        import librosa

        stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_chroma=n_chroma)
        return np.asarray(chroma, dtype=np.float32)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"Chroma extraction failed: {error}") from error


def extract_melspectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Extract Mel-spectrogram.

    Returns array of shape (n_mels, time_frames) as float32 (power scale).
    """
    audio = _ensure_mono_float32(audio)
    if audio.size == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)
    try:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        return np.asarray(mel, dtype=np.float32)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"Mel-spectrogram extraction failed: {error}") from error


def combine_features(
    mfcc: Optional[np.ndarray] = None,
    chroma: Optional[np.ndarray] = None,
    melspec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Combine feature matrices along the feature axis.

    Inputs should be shaped (features, time). Time dimensions must match for
    non-empty inputs. Returns a single array of shape (total_features, time).
    """
    matrices = [m for m in [mfcc, chroma, melspec] if m is not None]
    if not matrices:
        return np.zeros((0, 0), dtype=np.float32)

    # Validate time dimension consistency
    time_dims = [m.shape[1] for m in matrices]
    if len(set(time_dims)) > 1:
        raise ValueError("All feature inputs must have the same number of time frames")

    combined = np.concatenate(matrices, axis=0).astype(np.float32)
    return combined


__all__ = [
    "extract_mfcc",
    "extract_chroma",
    "extract_melspectrogram",
    "combine_features",
]

