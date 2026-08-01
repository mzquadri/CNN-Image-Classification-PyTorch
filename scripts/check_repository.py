"""Verify the versioned source and reference artifacts for this experiment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "requirements.txt",
    "src/model.py",
    "src/train.py",
    "src/train_fast.py",
    "results/test_results.json",
    "results/classification_report.txt",
    "results/figures/01_training_curves.png",
)


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))
    print(f"Repository check passed: {len(REQUIRED)} required artifacts available.")


if __name__ == "__main__":
    main()
