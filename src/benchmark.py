"""Reproduce the reference CIFAR-10 run and put baselines beside it.

    python -m src.benchmark --track cheap        # a couple of minutes, no GPU needed
    python -m src.benchmark --track cnn          # the reference configuration
    python -m src.benchmark --track pretrained   # the same budget, ImageNet weights
    python -m src.benchmark                      # all three

The repository reported 64.26 percent on CIFAR-10 from a custom CNN trained on
15,000 images for 12 epochs, and described that as competitive without relying on
pretrained models. Competitive against what was not stated: there was nothing to
compare it with. This adds the missing ends of the scale.

Everything is fitted on the same 13,500 training images and scored on the full
published 10,000-image test split. The reference configuration is reproduced
exactly as it was recorded, with one change: every source of randomness is seeded.
`src/train_fast.py` seeded only the two split generators, so model initialisation,
dropout and augmentation were not reproducible even on the same machine.

Writes results/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark.json"
DATA = ROOT / "data"

SEED = 42
SUBSET_SIZE = 15_000
VALIDATION_FRACTION = 0.1

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

#: The reference configuration, taken from results/hyperparameters.json. Fixed
#: before any run here and not revisited afterwards.
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-4

#: Published CIFAR-10 channel statistics. Constants, not fitted on any split, so
#: no information crosses from test into the normalisation.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

#: The pretrained arm sees the images at this resolution. ImageNet weights expect
#: 224; 96 keeps the comparison affordable while still letting the pretrained
#: filters see something other than a 32 pixel thumbnail. Stated because it is a
#: real handicap on that arm and the number should be read with it in mind.
#:
#: The upsampling happens on the device, on whole batches, not in the dataset
#: transform. Resizing each image on the CPU with a single worker took longer per
#: epoch than the training step it was feeding.
PRETRAINED_RESOLUTION = 96


def device():
    import torch
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = SEED) -> None:
    """Seed every generator the training path draws from.

    src/train_fast.py seeded the subset and the train/validation split, which
    fixes which images are used, and nothing else. Weight initialisation, dropout
    masks and augmentation draws all came from the global generator, so two runs
    of that script on one machine did not agree.
    """
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def splits():
    """The reference splits, rebuilt exactly.

    The validation set is wrapped around a second dataset object carrying the
    evaluation transform, so the augmentation applied to training never reaches
    it. That was already correct in src/train_fast.py and is kept.
    """
    import torch
    from torch.utils.data import Subset, random_split
    from torchvision import datasets, transforms

    train_steps = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    evaluation_steps = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

    augmented = datasets.CIFAR10(root=str(DATA), train=True, download=True,
                                 transform=transforms.Compose(train_steps))
    clean = datasets.CIFAR10(root=str(DATA), train=True, download=False,
                             transform=transforms.Compose(evaluation_steps))
    test = datasets.CIFAR10(root=str(DATA), train=False, download=True,
                            transform=transforms.Compose(evaluation_steps))

    chosen = torch.randperm(
        len(augmented), generator=torch.Generator().manual_seed(SEED)
    )[:SUBSET_SIZE].tolist()
    train_count = int((1 - VALIDATION_FRACTION) * SUBSET_SIZE)
    train_part, validation_part = random_split(
        Subset(augmented, chosen), [train_count, SUBSET_SIZE - train_count],
        generator=torch.Generator().manual_seed(SEED))

    validation = Subset(clean, [chosen[i] for i in validation_part.indices])
    return train_part, validation, test, [chosen[i] for i in train_part.indices]


def scores(actual, predicted) -> dict:
    """Top-1 accuracy, macro F1, and per-class precision, recall and F1.

    Accuracy alone is not enough even on a balanced test set. It equals macro
    recall here, so it says nothing about precision, and the reference run's own
    classification report shows classes ranging from 0.40 to 0.80 F1.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, per_class, support = precision_recall_fscore_support(
        actual, predicted, labels=range(len(CLASSES)), zero_division=0)
    return {
        "top1_accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(per_class[i]), "support": int(support[i])}
            for i, name in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=range(len(CLASSES))).tolist(),
        "n": len(actual),
    }


def confusion_pairs(matrix, top: int = 5) -> list[dict]:
    """The off-diagonal cells accounting for the most errors."""
    pairs = [{"true": CLASSES[r], "predicted": CLASSES[c], "count": int(matrix[r][c])}
             for r in range(len(CLASSES)) for c in range(len(CLASSES)) if r != c]
    pairs.sort(key=lambda e: -e["count"])
    total = sum(e["count"] for e in pairs)
    for entry in pairs:
        entry["share_of_errors"] = round(entry["count"] / total, 4) if total else 0.0
    return pairs[:top]


def upsample(images, resolution: int | None):
    """Resize a batch on the device it already lives on."""
    if resolution is None:
        return images
    import torch.nn.functional as functional
    return functional.interpolate(images, size=(resolution, resolution),
                                  mode="bilinear", align_corners=False)


def evaluate(model, loader, dev, resolution: int | None = None):
    import torch
    model.eval()
    predicted, actual = [], []
    with torch.no_grad():
        for images, labels in loader:
            out = model(upsample(images.to(dev), resolution))
            predicted.append(out.argmax(1).cpu().numpy())
            actual.append(labels.numpy())
    return np.concatenate(actual), np.concatenate(predicted)


def train_network(model, train_part, validation, test, dev, label: str,
                  resolution: int | None = None) -> dict:
    """One training run at the reference budget, choosing the epoch on validation."""
    import torch
    from torch.utils.data import DataLoader

    loaders = {
        "train": DataLoader(train_part, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0),
        "validation": DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0),
        "test": DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }
    model = model.to(dev)
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=LEARNING_RATE, epochs=EPOCHS,
        steps_per_epoch=len(loaders["train"]))

    history, best_state, best = [], None, -1.0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen, correct = 0.0, 0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(dev), labels.to(dev)
            optimiser.zero_grad()
            out = model(upsample(images, resolution))
            loss = criterion(out, labels)
            loss.backward()
            optimiser.step()
            schedule.step()
            running += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            seen += len(labels)

        actual, predicted = evaluate(model, loaders["validation"], dev, resolution)
        validation_accuracy = float((actual == predicted).mean())
        history.append({
            "epoch": epoch,
            "train_loss": running / seen,
            "train_accuracy": correct / seen,
            "validation_accuracy": validation_accuracy,
            "learning_rate": schedule.get_last_lr()[0],
        })
        print(f"    epoch {epoch:>2}  train {correct / seen * 100:5.2f}  "
              f"validation {validation_accuracy * 100:5.2f}", flush=True)

        # The epoch is chosen on validation. Test is scored once, afterwards.
        if validation_accuracy > best:
            best = validation_accuracy
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    model.to(dev)
    started = time.perf_counter()
    actual, predicted = evaluate(model, loaders["test"], dev, resolution)
    inference_seconds = time.perf_counter() - started

    result = scores(actual, predicted)
    return {
        "label": label,
        "history": history,
        "best_validation_accuracy": best,
        "epoch_chosen_on_validation": best_epoch,
        "test": result,
        "confusion_pairs": confusion_pairs(result["confusion_matrix"]),
        "train_seconds": round(train_seconds, 1),
        "images_per_second_at_inference": round(len(actual) / inference_seconds, 1),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "predictions": {"test_true": actual.tolist(),
                        "test_predicted": predicted.tolist()},
    }


def cheap_baselines(train_indices) -> dict:
    """What the picture looks like without a network.

    Two floors. Predicting one class is the level any accuracy has to be read
    against, and a linear model on raw pixels is the level at which the
    convolutional structure starts paying for itself.
    """
    from sklearn.linear_model import LogisticRegression
    from torchvision import datasets

    train = datasets.CIFAR10(root=str(DATA), train=True, download=False)
    test = datasets.CIFAR10(root=str(DATA), train=False, download=False)

    x_train = train.data[train_indices].reshape(len(train_indices), -1) / 255.0
    y_train = np.array(train.targets)[train_indices]
    x_test = test.data.reshape(len(test.data), -1) / 255.0
    y_test = np.array(test.targets)

    results = {}
    majority = int(np.bincount(y_train).argmax())
    results["most_frequent_class"] = {
        "label": f"always predict {CLASSES[majority]}",
        "test": scores(y_test, np.full(len(y_test), majority)),
        "train_seconds": 0.0,
        "parameters": 0,
    }

    print("  fitting logistic regression on raw pixels", flush=True)
    started = time.perf_counter()
    model = LogisticRegression(max_iter=300, C=0.01)
    model.fit(x_train, y_train)
    seconds = time.perf_counter() - started
    predicted = model.predict(x_test)
    results["logistic_regression_pixels"] = {
        "label": "logistic regression on raw pixels",
        "test": scores(y_test, predicted),
        "train_seconds": round(seconds, 1),
        "parameters": int(model.coef_.size + model.intercept_.size),
        "predictions": {"test_true": y_test.tolist(),
                        "test_predicted": predicted.tolist()},
    }
    for name, entry in results.items():
        entry["confusion_pairs"] = confusion_pairs(entry["test"]["confusion_matrix"])
        print(f"  {name:<28} top-1 {entry['test']['top1_accuracy'] * 100:5.2f}  "
              f"macro F1 {entry['test']['macro_f1'] * 100:5.2f}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["cheap", "cnn", "pretrained", "all"],
                        default="all")
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    import torch
    payload = {}
    if arguments.out.exists():
        payload = json.loads(arguments.out.read_text(encoding="utf-8"))

    dev = device()
    seed_everything()
    train_part, validation, test, train_indices = splits()
    print(f"  device {dev}")
    print(f"  train {len(train_part):,}   validation {len(validation):,}   "
          f"test {len(test):,}")

    payload["environment"] = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(dev),
        "device_name": (torch.xpu.get_device_name(0) if dev.type == "xpu"
                        else torch.cuda.get_device_name(0) if dev.type == "cuda"
                        else platform.processor() or "cpu"),
    }
    payload["data"] = {
        "dataset": "CIFAR-10, torchvision, published splits",
        "classes": list(CLASSES),
        "subset_of_train_used": SUBSET_SIZE,
        "train": len(train_part),
        "validation": len(validation),
        "test": len(test),
        "seed": SEED,
        "note": "the validation split carries the evaluation transform, not the "
                "training augmentation; the test split is the full published "
                "10,000 images and is scored once per model",
    }
    payload["configuration"] = {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "optimizer": "Adam",
        "weight_decay": WEIGHT_DECAY, "scheduler": "OneCycleLR",
        "augmentation": ["RandomCrop(32, padding=4)", "RandomHorizontalFlip",
                         "ColorJitter(brightness=0.2, contrast=0.2)"],
        "normalisation": {"mean": list(MEAN), "std": list(STD),
                          "source": "published CIFAR-10 constants, not fitted"},
        "source": "results/hyperparameters.json, the reference run",
    }

    models = payload.get("models", {})

    if arguments.track in ("cheap", "all"):
        print("\n  baselines without a network")
        models.update(cheap_baselines(train_indices))

    if arguments.track in ("cnn", "all"):
        print("\n  the reference CNN, reproduced")
        from .model import CIFAR10CNN
        seed_everything()
        models["custom_cnn"] = train_network(
            CIFAR10CNN(), train_part, validation, test, dev,
            "custom CNN, trained from scratch")

    if arguments.track in ("pretrained", "all"):
        print(f"\n  ResNet-18 with ImageNet weights, images at "
              f"{PRETRAINED_RESOLUTION} pixels")
        from torchvision import models as tv
        seed_everything()
        network = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        network.fc = torch.nn.Linear(network.fc.in_features, len(CLASSES))
        models["pretrained_resnet18"] = train_network(
            network, train_part, validation, test, dev,
            f"ResNet-18, ImageNet weights, {PRETRAINED_RESOLUTION}px",
            resolution=PRETRAINED_RESOLUTION)
        models["pretrained_resnet18"]["input_resolution"] = PRETRAINED_RESOLUTION

    payload["models"] = models

    # The original run's committed artifacts, read back and compared with the
    # reproduction. The point of keeping both is that the published figure can be
    # checked rather than trusted, and the difference between them stated.
    reference_path = ROOT / "results" / "test_results.json"
    if reference_path.is_file() and "custom_cnn" in models:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reproduced = models["custom_cnn"]["test"]["top1_accuracy"] * 100
        payload["reference_run"] = {
            "source": "results/test_results.json, committed before this audit",
            "test_accuracy": reference["test_accuracy"],
            "best_val_accuracy": reference["best_val_accuracy"],
            "n_parameters": reference["n_parameters"],
            "training_time_s": reference["training_time_s"],
            "hardware": "NVIDIA MX330, per the original README",
            "reproduced_test_accuracy": round(reproduced, 2),
            "difference_points": round(reproduced - reference["test_accuracy"], 2),
            "parameters_match": (reference["n_parameters"]
                                 == models["custom_cnn"]["parameters"]),
            "note": "the original script seeded only the two split generators, so "
                    "its exact figure was not reproducible even on its own machine; "
                    "this reproduction seeds everything and runs on different "
                    "hardware and a different torch build",
        }
        print(f"\n  reference run reported {reference['test_accuracy']:.2f}, "
              f"reproduced {reproduced:.2f} "
              f"({payload['reference_run']['difference_points']:+.2f} points)")

    print("\n  top-1 accuracy on the published test split")
    for name, entry in sorted(models.items(),
                              key=lambda kv: kv[1]["test"]["top1_accuracy"]):
        print(f"    {name:<26} {entry['test']['top1_accuracy'] * 100:6.2f}   "
              f"macro F1 {entry['test']['macro_f1'] * 100:6.2f}   "
              f"{entry['parameters']:>10,} parameters")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    print(f"\n  wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
