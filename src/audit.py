"""Audit the CIFAR-10 splits before trusting any accuracy computed on them.

    python -m src.audit

Three questions, asked in the order that matters.

Does the same image appear in both splits? A test image the model has trained on
is not held out, and nothing in a reported accuracy would say so.

Does a near-duplicate appear? Exact matching finds nothing in CIFAR-10, which is
not the same as finding nothing. A duplicate that has been flipped is byte
different and pixel identical to the eye, so the search is run twice, once
against the training images and once against their mirror images.

Are the classes balanced? CIFAR-10's published splits are, but the 15,000 image
subset the reference run trained on was drawn at random, so its balance is a
property of the draw rather than of the dataset.

Writes the findings into results/benchmark.json.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark.json"
DATA = ROOT / "data"

#: Distances are L2 over pixels scaled to [0, 1], across all 3,072 dimensions.
#: A pair at 1.0 differs by 0.018 per channel on average, which is far closer
#: than any two distinct photographs get.
THRESHOLDS = (1.0, 2.0, 3.0, 4.0)


def content_hashes(images) -> list[str]:
    return [hashlib.sha256(image.tobytes()).hexdigest() for image in images]


def nearest_neighbour_distances(test_images, train_images, include_flips: bool):
    """Distance from each test image to the closest training image.

    Done as a matrix product on whatever device is available, because the naive
    form is 10,000 by 50,000 by 3,072 and matters more than it costs.
    """
    import torch

    from .benchmark import device

    dev = device()
    train = torch.tensor(train_images, dtype=torch.float32, device=dev) / 255.0
    if include_flips:
        train = torch.cat([train, torch.flip(train, dims=[2])])
    train = train.reshape(len(train), -1)
    test = (torch.tensor(test_images, dtype=torch.float32, device=dev) / 255.0
            ).reshape(len(test_images), -1)

    train_norms = (train * train).sum(1)
    best = torch.full((len(test),), float("inf"), device=dev)
    argument = torch.zeros(len(test), dtype=torch.long, device=dev)
    for start in range(0, len(test), 400):
        chunk = test[start:start + 400]
        # The expanded form is fast but loses precision exactly where it matters.
        # Each squared norm is around 1,000, so in float32 their difference carries
        # an absolute error near 1e-4, and a genuinely identical pair comes back at
        # roughly 0.01 instead of 0. It is used only to rank candidates.
        distance = ((chunk * chunk).sum(1, keepdim=True) + train_norms.unsqueeze(0)
                    - 2.0 * (chunk @ train.T))
        value, index = distance.clamp_min(0).min(1)
        best[start:start + 400] = value.sqrt()
        argument[start:start + 400] = index

    # Recompute each winning pair by direct subtraction in double precision. The
    # ranking is unchanged; the distances become exact, which is what lets a
    # reported closest pair be read as a real number rather than a floor.
    exact = torch.empty(len(test), dtype=torch.float64, device=dev)
    for start in range(0, len(test), 400):
        rows = test[start:start + 400].double()
        partners = train[argument[start:start + 400]].double()
        exact[start:start + 400] = (rows - partners).pow(2).sum(1).sqrt()

    return exact.cpu().numpy(), (argument % len(train_images)).cpu().numpy()


def run() -> dict:
    import torch
    from torchvision import datasets

    from .benchmark import CLASSES, SEED, SUBSET_SIZE

    train = datasets.CIFAR10(root=str(DATA), train=True, download=True)
    test = datasets.CIFAR10(root=str(DATA), train=False, download=True)
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)

    train_hashes = content_hashes(train.data)
    test_hashes = content_hashes(test.data)
    known = set(train_hashes)
    exact = {
        "duplicate_images_within_train": len(train_hashes) - len(known),
        "duplicate_images_within_test": len(test_hashes) - len(set(test_hashes)),
        "test_images_identical_to_a_train_image": sum(
            1 for value in test_hashes if value in known),
    }
    print(f"  exact duplicates within train {exact['duplicate_images_within_train']}, "
          f"within test {exact['duplicate_images_within_test']}, "
          f"across splits {exact['test_images_identical_to_a_train_image']}")

    near = {}
    for name, flips in (("pixels", False), ("pixels_or_mirror_image", True)):
        distances, indices = nearest_neighbour_distances(test.data, train.data, flips)
        counts = {str(t): int((distances < t).sum()) for t in THRESHOLDS}
        closest = int(distances.argmin())
        near[name] = {
            "closest_pair_distance": round(float(distances.min()), 4),
            "closest_pair_mean_channel_difference": round(
                float(distances.min()) / np.sqrt(3072), 5),
            "median_distance": round(float(np.median(distances)), 4),
            "test_images_closer_than": counts,
            "fraction_closer_than_1": round(counts["1.0"] / len(distances), 5),
            "closest_pair": {
                "test_class": CLASSES[test_labels[closest]],
                "train_class": CLASSES[train_labels[indices[closest]]],
                "labels_agree": bool(
                    test_labels[closest] == train_labels[indices[closest]]),
            },
        }
        print(f"  {name}: closest pair {near[name]['closest_pair_distance']}, "
              f"{counts['1.0']} test images within 1.0, {counts['3.0']} within 3.0")

    generator = torch.Generator().manual_seed(SEED)
    chosen = torch.randperm(len(train.data), generator=generator)[:SUBSET_SIZE].tolist()
    subset = collections.Counter(train_labels[chosen])
    per_class = [int(subset[i]) for i in range(len(CLASSES))]
    balance = {
        "published_train_per_class": min(
            collections.Counter(train_labels).values()),
        "published_test_per_class": min(
            collections.Counter(test_labels).values()),
        "subset_per_class": dict(zip(CLASSES, per_class, strict=True)),
        "subset_smallest_class": min(per_class),
        "subset_largest_class": max(per_class),
        "subset_spread": max(per_class) - min(per_class),
        "subset_relative_spread": round(
            (max(per_class) - min(per_class)) / (sum(per_class) / len(per_class)), 4),
    }
    print(f"  the seeded 15,000 subset spans {balance['subset_smallest_class']} to "
          f"{balance['subset_largest_class']} images per class, "
          f"{balance['subset_relative_spread'] * 100:.1f} percent of the mean")

    return {
        "exact_duplicates": exact,
        "near_duplicates": near,
        "class_balance": balance,
        "method": "content hashing for exact matches; nearest neighbour in L2 over "
                  "pixels scaled to [0, 1] for near matches, run once against the "
                  "training images and once against their mirror images",
        "limits": "this covers identical and mirrored images only. A duplicate that "
                  "has been cropped, rescaled or recompressed will not be found by "
                  "it, and a published analysis using a richer search reports a "
                  "higher figure, so what is measured here is a lower bound",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    payload = {}
    if arguments.out.exists():
        payload = json.loads(arguments.out.read_text(encoding="utf-8"))
    payload["audit"] = run()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    print(f"\n  wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
