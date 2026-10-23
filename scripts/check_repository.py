"""Fail if the README stops agreeing with the recorded results.

    python scripts/check_repository.py

Checks that the files the README points at exist and compile, and that every
number it quotes still matches `results/benchmark.json`.

The second is what matters here. Training needs a GPU and is not repeated in
continuous integration, so the committed results file is the only record of it. A
README drifting from that file would be the only description of runs nobody is
going to redo.

Each claim is matched with its surrounding words included, so a value that has
moved into a different sentence does not accidentally satisfy a check.
"""

from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark.json"
REFERENCE = ROOT / "results" / "test_results.json"
DIAGRAM = ROOT / "docs" / "diagrams" / "architecture.svg"

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "src/model.py",
    "src/benchmark.py",
    "src/audit.py",
    "src/train.py",
    "src/train_fast.py",
    "tests/test_data.py",
    "tests/test_metrics.py",
    "scripts/figures/generate_figures.py",
    "docs/diagrams/architecture.svg",
    "docs/figures/01_model_comparison.png",
    "docs/figures/02_per_class.png",
    "docs/figures/03_confusion.png",
    "docs/figures/04_failure_examples.png",
    "results/benchmark.json",
    "results/test_results.json",
    "results/hyperparameters.json",
)

PRETTY = {
    "pretrained_resnet18": "ResNet-18, ImageNet weights",
    "custom_cnn": "Custom CNN, from scratch",
    "logistic_regression_pixels": "Logistic regression on raw pixels",
    "most_frequent_class": "Always predict one class",
}


def claims(data: dict) -> list[tuple[str, str]]:
    """Every quoted number, paired with enough context to anchor it."""
    models = data["models"]
    audit = data["audit"]
    reference = data["reference_run"]
    cnn = models["custom_cnn"]
    pretrained = models["pretrained_resnet18"]
    linear = models["logistic_regression_pixels"]
    plain = audit["near_duplicates"]["pixels"]
    mirrored = audit["near_duplicates"]["pixels_or_mirror_image"]

    def percent(entry, key="top1_accuracy"):
        return f"{entry['test'][key] * 100:.2f}"

    out = [
        # The headline and the reproduction.
        (f"reaches {percent(cnn)} percent on the published test split",
         "the reproduced accuracy"),
        (f"recorded {reference['test_accuracy']:.2f} percent",
         "the originally reported accuracy"),
        (f"reaches {percent(pretrained)} percent under the identical budget",
         "the pretrained accuracy"),
        (f"| Test accuracy | {reference['test_accuracy']:.2f} | {percent(cnn)} |",
         "the reproduction table accuracy row"),
        (f"| Best validation accuracy | {reference['best_val_accuracy']:.2f} | "
         f"{cnn['best_validation_accuracy'] * 100:.2f} |",
         "the reproduction table validation row"),
        (f"| Parameters | {reference['n_parameters']:,} | "
         f"{cnn['parameters']:,} |", "the reproduction table parameter row"),
        (f"differs by {reference['difference_points']:.2f} points",
         "the reproduction difference"),

        # Splits.
        (f"| train | {data['data']['train']:,} | fitting |", "the train split row"),
        (f"| validation | {data['data']['validation']:,} |",
         "the validation split row"),
        (f"| test | {data['data']['test']:,} |", "the test split row"),
        (f"subset of {data['data']['subset_of_train_used']:,} training images",
         "the subset size"),

        # Audit.
        (f"| Byte-identical duplicates within train | "
         f"{audit['exact_duplicates']['duplicate_images_within_train']} |",
         "the train duplicate row"),
        (f"| Test images byte-identical to a training image | "
         f"{audit['exact_duplicates']['test_images_identical_to_a_train_image']} |",
         "the cross-split duplicate row"),
        (f"| Test images within L2 1.0 of a training image | "
         f"{plain['test_images_closer_than']['1.0']} |",
         "the pixel near-duplicate row"),
        (f"or its mirror | {mirrored['test_images_closer_than']['1.0']} |",
         "the mirrored near-duplicate row"),
        (f"closest pair at\n"
         f"{audit['near_duplicates']['pixels_or_mirror_image']['closest_pair_distance']:.4f}",
         "the closest mirrored pair"),
        (f"mean channel difference of "
         f"{audit['near_duplicates']['pixels_or_mirror_image']['closest_pair_mean_channel_difference']:.4f}",
         "the closest pair channel difference"),
        ("and thirteen within 3.0", "the count within 3.0")
        if mirrored["test_images_closer_than"]["3.0"] == 13
        else (f"{mirrored['test_images_closer_than']['3.0']} within 3.0",
              "the count within 3.0"),
        (f"median test image sits {mirrored['median_distance']:.2f}",
         "the median neighbour distance"),
        (f"spans {audit['class_balance']['subset_smallest_class']:,} to "
         f"{audit['class_balance']['subset_largest_class']:,} images per class",
         "the subset balance range"),
        (f"spread of {audit['class_balance']['subset_relative_spread'] * 100:.1f} percent",
         "the subset balance spread"),
        (f"{audit['class_balance']['published_train_per_class']:,} training and "
         f"{audit['class_balance']['published_test_per_class']:,} test images",
         "the published per-class counts"),
    ]

    for name, entry in models.items():
        out.append((
            f"| {PRETTY[name]} | {percent(entry)} | {percent(entry, 'macro_f1')} | "
            f"{entry['parameters']:,} |", f"the {name} results row"))

    gain_over_linear = (cnn["test"]["top1_accuracy"]
                        - linear["test"]["top1_accuracy"]) * 100
    gain_from_pretraining = (pretrained["test"]["top1_accuracy"]
                             - cnn["test"]["top1_accuracy"]) * 100
    out += [
        (f"worth {gain_over_linear:.2f} points over a linear model",
         "the gain over the linear baseline"),
        (f"worth a further\n{gain_from_pretraining:.2f} points",
         "the gain from pretraining"),
        (f"{pretrained['parameters'] / cnn['parameters']:.1f} times the\nparameters",
         "the parameter multiple"),
        (f"{pretrained['train_seconds'] / cnn['train_seconds']:.1f} times the fit time",
         "the fit time multiple"),
        (f"{cnn['images_per_second_at_inference']
            / pretrained['images_per_second_at_inference']:.1f} times slower "
         f"inference", "the inference multiple"),
        (f"custom network takes {cnn['train_seconds']:.1f} seconds",
         "the CNN training time"),
        (f"pretrained one {pretrained['train_seconds']:.1f} seconds",
         "the pretrained training time"),
    ]

    # Confusion and per-class.
    worst_cnn = cnn["confusion_pairs"][0]
    worst_pre = pretrained["confusion_pairs"][0]
    out += [
        (f"{worst_cnn['count']} images for the\ncustom CNN and "
         f"{worst_pre['count']} for the pretrained one", "the cat and dog counts"),
        (f"from {worst_cnn['share_of_errors'] * 100:.1f} percent of the custom "
         f"network's errors to {worst_pre['share_of_errors'] * 100:.1f} percent",
         "the shares of the cat and dog cell"),
    ]

    deltas = {c: (pretrained["test"]["per_class"][c]["f1"]
                  - cnn["test"]["per_class"][c]["f1"]) * 100
              for c in data["data"]["classes"]}
    vehicles = {"airplane", "automobile", "ship", "truck"}
    animal = {c: v for c, v in deltas.items() if c not in vehicles}
    vehicle = {c: v for c, v in deltas.items() if c in vehicles}
    out += [
        (f"between {min(vehicle.values()):.1f} and {max(vehicle.values()):.1f}\n"
         f"F1 points", "the vehicle gain range"),
        (f"between {min(animal.values()):.1f} and {max(animal.values()):.1f}",
         "the animal gain range"),
        (f"Bird gains {deltas['bird']:.1f} and\ncat {deltas['cat']:.1f}",
         "the bird and cat gains"),
    ]
    return out


def diagram_claims(data: dict, reference: dict) -> list[tuple[str, str]]:
    """The diagram quotes numbers too, and nothing used to check them.

    `docs/diagrams/architecture.svg` is hand-written, so its figures are a second
    copy of values that live in the results files. Until this check existed, the
    only assertion about it was that the file is present. A second copy nobody
    compares is how a diagram comes to contradict the README beside it.

    It is also the one place in the repository where both runs appear together, so
    each badge has to say which run it is describing.
    """
    cnn = data["models"]["custom_cnn"]
    config = data["configuration"]
    per_class = reference["per_class_accuracy"]
    best = max(per_class, key=lambda cls: per_class[cls])
    worst = min(per_class, key=lambda cls: per_class[cls])
    reproduced = cnn["test"]["top1_accuracy"] * 100

    return [
        (f"{cnn['parameters']:,} trainable parameters", "the parameter count in its subtitle"),
        (f"Test accuracy {reference['test_accuracy']:.2f}%", "the original run's accuracy badge"),
        (f"Seeded reproduction {reproduced:.2f}%", "the reproduction badge"),
        (f"reproduces to {reproduced:.2f}%", "the reproduced figure in the strip's label"),
        (f"Best: {best} {per_class[best]:.1f}%", "the best class and its accuracy"),
        (f"Hardest: {worst} {per_class[worst]:.1f}%", "the hardest class and its accuracy"),
        (f"{config['epochs']} epochs", "the epoch count"),
        (f"batch {config['batch_size']}", "the batch size"),
        (f"OneCycleLR max {config['learning_rate']}", "the schedule and its peak learning rate"),
        ("results/test_results.json", "which file the badges came from"),
    ]


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"  missing required files: {', '.join(missing)}")

    for source in sorted((ROOT / "src").glob("*.py")):
        py_compile.compile(source, doraise=True)
    print(f"  {len(REQUIRED_FILES)} required files present, src/ compiles")

    if not RESULTS.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    # The parameter count is quoted in the README and is checkable directly.
    import sys
    sys.path.insert(0, str(ROOT))
    from src.model import CIFAR10CNN
    counted = sum(p.numel() for p in CIFAR10CNN().parameters())
    recorded = data["models"]["custom_cnn"]["parameters"]
    if counted != recorded:
        raise SystemExit(f"  the model has {counted:,} parameters but the results "
                         f"record {recorded:,}")
    print(f"  the model definition has {counted:,} parameters, matching the results")

    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    failures = []
    checks = claims(data)
    for expected, description in checks:
        if re.sub(r"\s+", " ", expected) not in flat:
            failures.append(f"  README does not state {description}: expected "
                            f"{re.sub(r'\\s+', ' ', expected).strip()!r}")
    print(f"  {len(checks) - len(failures)} of {len(checks)} recorded numbers "
          f"found in the README")

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    svg = re.sub(r"\s+", " ", DIAGRAM.read_text(encoding="utf-8"))
    diagram = diagram_claims(data, reference)
    diagram_failures = [
        f"  the diagram does not state {description}: expected "
        f"{re.sub(r'\s+', ' ', expected).strip()!r}"
        for expected, description in diagram
        if re.sub(r"\s+", " ", expected) not in svg
    ]
    print(f"  {len(diagram) - len(diagram_failures)} of {len(diagram)} recorded numbers "
          f"found in the diagram")
    failures += diagram_failures

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} claims no longer match the results files")

    print("  README, diagram and the results files agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
