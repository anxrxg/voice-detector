"""Project entry point.

Phase 1 helper: Validate dataset directory structure.

Usage:
    python main.py --check-datasets

This command is read-only and adheres to the guardrails in `llm.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from utils.preprocessing import validate_dataset_structure


def run_dataset_check() -> int:
    project_root = Path(__file__).parent
    status = validate_dataset_structure(project_root)

    print("Project root:", project_root)
    print("Exists:", status.exists)
    print("All required directories present:", status.required_directories_present)
    if status.missing_directories:
        print("Missing directories:")
        for missing in status.missing_directories:
            print(" -", missing)
    if status.notes:
        print("Notes:")
        for note in status.notes:
            print(" -", note)

    return 0 if status.required_directories_present else 1


def main() -> NoReturn:
    parser = argparse.ArgumentParser(description="Age & Emotion Voice Detector – Utilities")
    parser.add_argument(
        "--check-datasets",
        action="store_true",
        help="Validate that required dataset directories exist",
    )
    args = parser.parse_args()

    if args.check_datasets:
        raise SystemExit(run_dataset_check())

    parser.print_help()
    raise SystemExit(0)


if __name__ == "__main__":
    main()

