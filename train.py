"""Unified training script for gender, age, and emotion models.

Usage examples:
  - Train everything with defaults (project root = this file's directory):
      python train.py --all

  - Train only gender and emotion:
      python train.py --task gender --task emotion

  - Specify project root (e.g., when running in Colab/Drive):
      python train.py --project-root /content/drive/MyDrive/voice-detector --all

  - Provide label CSVs:
      python train.py --all \
        --age-labels data/age/age_labels.csv \
        --emotion-labels data/emotion/emotion_labels.csv

Models are saved to `<project_root>/models` by default:
  - gender_model.pkl
  - age_model.pkl
  - emotion_model.pkl
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.preprocessing import (
    validate_dataset_structure,
    preprocess_audio_file,
    prepare_gender_splits,
    prepare_age_splits,
    prepare_emotion_splits,
)
from utils.model_utils import (
    make_feature_vector_from_audio,
    train_gender_model,
    evaluate_classification,
    train_age_model,
    evaluate_regression,
    train_emotion_model,
    save_sklearn_model,
)


@dataclass(frozen=True)
class SplitMatrices:
    X_train: np.ndarray
    y_train: Sequence
    X_val: np.ndarray
    y_val: Sequence
    X_test: np.ndarray
    y_test: Sequence


def _compute_feature_vector(audio_path: str) -> Optional[np.ndarray]:
    """Preprocess + extract compact feature vector for a single file."""
    try:
        audio, sr = preprocess_audio_file(audio_path, save=False)
        vec = make_feature_vector_from_audio(audio, sr)
        if vec is None or vec.size == 0:
            return None
        return vec.astype(np.float32)
    except Exception as error:  # noqa: BLE001
        print(f"[WARN] Feature extraction failed for {audio_path}: {error}")
        return None


def _build_matrix(
    entries: List[Tuple[str, Optional[object]]],
    num_workers: int = 1,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, List[object]]:
    """Compute feature matrix X and labels y for a list of (filepath, label).

    Skips entries with missing labels or failed feature extraction.
    """
    # Optionally truncate for quick iterations
    if limit is not None and limit > 0:
        trimmed: List[Tuple[str, Optional[object]]] = []
        for filepath, label in entries:
            if label is None:
                continue
            trimmed.append((filepath, label))
            if len(trimmed) >= int(limit):
                break
        entries = trimmed

    if not entries:
        return np.zeros((0, 0), dtype=np.float32), []

    # Parallel feature extraction
    max_workers = max(1, int(num_workers))
    features: List[np.ndarray] = []
    labels: List[object] = []

    if max_workers == 1:
        for filepath, label in entries:
            if label is None:
                continue
            vec = _compute_feature_vector(filepath)
            if vec is None:
                continue
            features.append(vec)
            labels.append(label)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {
                executor.submit(_compute_feature_vector, filepath): (filepath, label)
                for filepath, label in entries
                if label is not None
            }
            for future in as_completed(future_to_meta):
                filepath, label = future_to_meta[future]
                try:
                    vec = future.result()
                except Exception as error:  # noqa: BLE001
                    print(f"[WARN] Future failed for {filepath}: {error}")
                    continue
                if vec is None:
                    continue
                features.append(vec)
                labels.append(label)  # type: ignore[arg-type]

    if not features:
        return np.zeros((0, 0), dtype=np.float32), []

    # Stack into 2D matrix
    X = np.vstack([v.reshape(1, -1) for v in features]).astype(np.float32)
    y = list(labels)
    return X, y


def _prepare_and_vectorize_splits(
    split_dict: dict,
    num_workers: int,
    limit_per_split: Optional[int],
) -> SplitMatrices:
    X_train, y_train = _build_matrix(split_dict.get("train", []), num_workers, limit_per_split)
    X_val, y_val = _build_matrix(split_dict.get("val", []), num_workers, limit_per_split)
    X_test, y_test = _build_matrix(split_dict.get("test", []), num_workers, limit_per_split)
    return SplitMatrices(X_train, y_train, X_val, y_val, X_test, y_test)


def train_gender(
    project_root: Path,
    models_dir: Path,
    model_type: str,
    num_workers: int,
    limit_per_split: Optional[int],
) -> Optional[Path]:
    print("\n=== Training: Gender ===")
    splits = prepare_gender_splits(project_root)
    matrices = _prepare_and_vectorize_splits(splits, num_workers, limit_per_split)
    print("Gender shapes:", matrices.X_train.shape, matrices.X_val.shape, matrices.X_test.shape)

    if matrices.X_train.size == 0 or len(matrices.y_train) == 0:
        print("[INFO] No training data for gender. Skipping.")
        return None

    bundle = train_gender_model(matrices.X_train, matrices.y_train, model_type=model_type)

    if matrices.X_val.size and matrices.y_val:
        val_metrics = evaluate_classification(bundle, matrices.X_val, matrices.y_val)
        print(f"Gender acc (val): {val_metrics.accuracy:.4f}")
    if matrices.X_test.size and matrices.y_test:
        test_metrics = evaluate_classification(bundle, matrices.X_test, matrices.y_test)
        print(f"Gender acc (test): {test_metrics.accuracy:.4f}")

    out_path = models_dir / "gender_model.pkl"
    saved = save_sklearn_model(bundle, str(out_path))
    print("Saved:", saved)
    return Path(saved)


def train_age(
    project_root: Path,
    models_dir: Path,
    model_type: str,
    num_workers: int,
    limit_per_split: Optional[int],
    labels_csv: Optional[Path],
) -> Optional[Path]:
    print("\n=== Training: Age ===")
    splits = prepare_age_splits(project_root, labels_csv=labels_csv)
    matrices = _prepare_and_vectorize_splits(splits, num_workers, limit_per_split)
    print("Age shapes:", matrices.X_train.shape, matrices.X_val.shape, matrices.X_test.shape)

    if matrices.X_train.size == 0 or len(matrices.y_train) == 0:
        print("[INFO] No training data for age (ensure labels CSV). Skipping.")
        return None

    # y must be floats
    y_train = [float(v) for v in matrices.y_train]
    age_model = train_age_model(matrices.X_train, y_train, model_type=model_type)

    if matrices.X_val.size and matrices.y_val:
        y_val = [float(v) for v in matrices.y_val]
        val_metrics = evaluate_regression(age_model, matrices.X_val, y_val)
        print(f"Age MAE/RMSE (val): {val_metrics.mae:.4f} / {val_metrics.rmse:.4f}")
    if matrices.X_test.size and matrices.y_test:
        y_test = [float(v) for v in matrices.y_test]
        test_metrics = evaluate_regression(age_model, matrices.X_test, y_test)
        print(f"Age MAE/RMSE (test): {test_metrics.mae:.4f} / {test_metrics.rmse:.4f}")

    out_path = models_dir / "age_model.pkl"
    saved = save_sklearn_model(age_model, str(out_path))
    print("Saved:", saved)
    return Path(saved)


def train_emotion(
    project_root: Path,
    models_dir: Path,
    model_type: str,
    num_workers: int,
    limit_per_split: Optional[int],
    labels_csv: Optional[Path],
    valid_emotions: Optional[Sequence[str]],
) -> Optional[Path]:
    print("\n=== Training: Emotion ===")
    splits = prepare_emotion_splits(project_root, labels_csv=labels_csv)
    matrices = _prepare_and_vectorize_splits(splits, num_workers, limit_per_split)
    print("Emotion shapes:", matrices.X_train.shape, matrices.X_val.shape, matrices.X_test.shape)

    if matrices.X_train.size == 0 or len(matrices.y_train) == 0:
        print("[INFO] No training data for emotion (ensure labels CSV). Skipping.")
        return None

    bundle = train_emotion_model(
        matrices.X_train,
        matrices.y_train,
        valid_emotions=valid_emotions,
        model_type=model_type,
    )

    if matrices.X_val.size and matrices.y_val:
        val_metrics = evaluate_classification(bundle, matrices.X_val, matrices.y_val)
        print(f"Emotion acc (val): {val_metrics.accuracy:.4f}")
    if matrices.X_test.size and matrices.y_test:
        test_metrics = evaluate_classification(bundle, matrices.X_test, matrices.y_test)
        print(f"Emotion acc (test): {test_metrics.accuracy:.4f}")

    out_path = models_dir / "emotion_model.pkl"
    saved = save_sklearn_model(bundle, str(out_path))
    print("Saved:", saved)
    return Path(saved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train voice-detector models")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).parent),
        help="Project root (containing data/ and utils/)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to save trained models (default: <project_root>/models)",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=["gender", "age", "emotion"],
        help="Task(s) to train; may be specified multiple times",
    )
    parser.add_argument("--all", action="store_true", help="Train all tasks")

    # Optional labels CSVs
    parser.add_argument("--age-labels", type=str, default=None, help="CSV file with age labels")
    parser.add_argument(
        "--emotion-labels", type=str, default=None, help="CSV file with emotion labels"
    )
    parser.add_argument(
        "--valid-emotions",
        type=str,
        default=None,
        help="Comma-separated list of valid emotions (optional)",
    )

    # Model type knobs
    parser.add_argument(
        "--gender-model",
        type=str,
        choices=["linear_svc", "logreg"],
        default="linear_svc",
        help="Classifier type for gender",
    )
    parser.add_argument(
        "--age-model",
        type=str,
        choices=["rf", "svr"],
        default="rf",
        help="Regressor type for age",
    )
    parser.add_argument(
        "--emotion-model",
        type=str,
        choices=["logreg", "linear_svc"],
        default="logreg",
        help="Classifier type for emotion",
    )

    # Performance/iteration knobs
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel workers for feature extraction",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional max items per split for quick runs",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    models_dir = Path(args.models_dir).resolve() if args.models_dir else project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", project_root)
    print("Models dir:  ", models_dir)

    status = validate_dataset_structure(project_root)
    print("Datasets ready:", status.required_directories_present)
    if status.missing_directories:
        print("Missing required directories:")
        for m in status.missing_directories:
            print(" -", m)

    selected_tasks: List[str] = []
    if args.all or not args.task:
        selected_tasks = ["gender", "age", "emotion"]
    else:
        selected_tasks = list(dict.fromkeys(args.task))  # de-dup preserving order
    print("Selected tasks:", ", ".join(selected_tasks))

    age_labels_csv = Path(args.age_labels).resolve() if args.age_labels else None
    emotion_labels_csv = (
        Path(args.emotion_labels).resolve() if args.emotion_labels else None
    )
    valid_emotions: Optional[List[str]] = (
        [s.strip() for s in args.valid_emotions.split(",") if s.strip()]
        if args.valid_emotions
        else None
    )

    # Execute requested tasks
    if "gender" in selected_tasks:
        train_gender(
            project_root,
            models_dir,
            model_type=args.gender_model,
            num_workers=args.num_workers,
            limit_per_split=args.limit_per_split,
        )

    if "age" in selected_tasks:
        train_age(
            project_root,
            models_dir,
            model_type=args.age_model,
            num_workers=args.num_workers,
            limit_per_split=args.limit_per_split,
            labels_csv=age_labels_csv,
        )

    if "emotion" in selected_tasks:
        train_emotion(
            project_root,
            models_dir,
            model_type=args.emotion_model,
            num_workers=args.num_workers,
            limit_per_split=args.limit_per_split,
            labels_csv=emotion_labels_csv,
            valid_emotions=valid_emotions,
        )


if __name__ == "__main__":
    main()
