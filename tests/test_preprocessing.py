import os
from pathlib import Path

import numpy as np

from utils.preprocessing import (
    validate_dataset_structure,
    normalize_audio,
    reduce_noise,
)
from utils.feature_extraction import (
    extract_mfcc,
    extract_chroma,
    extract_melspectrogram,
    combine_features,
)
from utils.preprocessing import (
    split_list,
    prepare_gender_splits,
    prepare_age_splits,
    prepare_emotion_splits,
)


def test_validate_dataset_structure_success(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    # Create required directories
    (project_root / "data" / "gender" / "male").mkdir(parents=True)
    (project_root / "data" / "gender" / "female").mkdir(parents=True)
    (project_root / "data" / "age" / "wav").mkdir(parents=True)
    (project_root / "data" / "emotion" / "wav").mkdir(parents=True)

    status = validate_dataset_structure(project_root)
    assert status.exists is True
    assert status.required_directories_present is True
    assert status.missing_directories == []


def test_validate_dataset_structure_missing(tmp_path: Path):
    project_root = tmp_path
    # Create only some directories
    (project_root / "data" / "gender" / "male").mkdir(parents=True)

    status = validate_dataset_structure(project_root)
    assert status.exists is True
    assert status.required_directories_present is False
    assert "data/gender/female" in status.missing_directories
    assert "data/age/wav" in status.missing_directories
    assert "data/emotion/wav" in status.missing_directories


def test_validate_dataset_structure_invalid_arg():
    try:
        validate_dataset_structure("")
        assert False, "Expected ValueError for empty project_root"
    except ValueError:
        assert True


def test_normalize_audio_no_change_for_silent():
    silent = np.zeros(16000, dtype=np.float32)
    normalized = normalize_audio(silent)
    assert np.allclose(silent, normalized)


def test_normalize_audio_scales_peak_to_target():
    signal = np.array([0.2, -0.5, 0.3], dtype=np.float32)
    normalized = normalize_audio(signal, target_peak=0.5)
    assert np.isclose(np.max(np.abs(normalized)), 0.5, atol=1e-6)


def test_reduce_noise_handles_empty_input():
    empty = np.array([], dtype=np.float32)
    denoised = reduce_noise(empty, sample_rate=16000)
    assert denoised.size == 0


def _sine(sr=16000, seconds=0.1, freq=440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_extractors_return_expected_shapes():
    sr = 16000
    audio = _sine(sr=sr, seconds=0.2)

    mfcc = extract_mfcc(audio, sr, n_mfcc=13)
    chroma = extract_chroma(audio, sr, n_chroma=12)
    mel = extract_melspectrogram(audio, sr, n_mels=64)

    assert mfcc.ndim == 2 and mfcc.shape[0] == 13
    assert chroma.ndim == 2 and chroma.shape[0] == 12
    assert mel.ndim == 2 and mel.shape[0] == 64


def test_combine_features_validates_time_and_concatenates():
    # Create dummy matrices with same time dimension
    mfcc = np.zeros((13, 5), dtype=np.float32)
    chroma = np.zeros((12, 5), dtype=np.float32)
    mel = np.zeros((64, 5), dtype=np.float32)

    combined = combine_features(mfcc, chroma, mel)
    assert combined.shape == (13 + 12 + 64, 5)

    # Mismatched time dims should raise
    try:
        _ = combine_features(mfcc, np.zeros((12, 6), dtype=np.float32), mel)
        assert False, "Expected ValueError for mismatched time dimensions"
    except ValueError:
        pass


def test_split_list_deterministic_and_ratios():
    items = [Path(f"f{i}.wav") for i in range(10)]
    s1 = split_list(items, 0.6, 0.2, 0.2, seed=123)
    s2 = split_list(items, 0.6, 0.2, 0.2, seed=123)
    assert [p.name for p in s1["train"]] == [p.name for p in s2["train"]]
    assert len(s1["train"]) == 6
    assert len(s1["val"]) == 2
    assert len(s1["test"]) == 2


def test_prepare_gender_splits_handles_missing_dirs(tmp_path: Path):
    root = tmp_path
    (root / "data" / "gender" / "male").mkdir(parents=True)
    # Leave female empty/missing
    # Create dummy files
    (root / "data" / "gender" / "male" / "a.wav").write_bytes(b"\0\0")
    splits = prepare_gender_splits(root, extensions=("wav",))
    assert set(splits.keys()) == {"train", "val", "test"}


def test_prepare_age_emotion_splits_empty(tmp_path: Path):
    root = tmp_path
    (root / "data" / "age" / "wav").mkdir(parents=True)
    (root / "data" / "emotion" / "wav").mkdir(parents=True)
    age = prepare_age_splits(root, extensions=("wav",))
    emo = prepare_emotion_splits(root, extensions=("wav",))
    for d in (age, emo):
        assert set(d.keys()) == {"train", "val", "test"}

