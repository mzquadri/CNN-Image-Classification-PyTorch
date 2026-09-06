"""Render the README figures from results/benchmark.json.

    python scripts/figures/generate_figures.py

Each figure answers one question:

  01  how do the four models compare on the same held-out test split
  02  which classes the pretrained backbone actually helps
  03  what each network confuses, side by side on the same scale
  04  what the mistakes look like, chosen from the largest confusion cells

Everything is read from the recorded results rather than recomputed, except the
image grid in 04, which needs the pixels and reads the recorded predictions to
decide which pixels to show.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import portfolio_style as ps

RESULTS = ROOT / "results" / "benchmark.json"
FIGURES = ROOT / "docs" / "figures"

NICE = {
    "most_frequent_class": "Always one class",
    "logistic_regression_pixels": "Logistic regression\non raw pixels",
    "custom_cnn": "Custom CNN\nfrom scratch",
    "pretrained_resnet18": "ResNet-18\nImageNet weights",
}
ORDER = ("most_frequent_class", "logistic_regression_pixels", "custom_cnn",
         "pretrained_resnet18")


def load() -> dict:
    if not RESULTS.exists():
        raise SystemExit("run `python -m src.benchmark` first")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    missing = [name for name in ORDER if name not in data.get("models", {})]
    if missing:
        raise SystemExit(f"results/benchmark.json is missing {', '.join(missing)}; "
                         f"run `python -m src.benchmark`")
    return data


def figure_01_model_comparison(data: dict) -> None:
    """How do the four compare on the same held-out images?"""
    models = data["models"]
    accuracy = [models[n]["test"]["top1_accuracy"] * 100 for n in ORDER]
    macro = [models[n]["test"]["macro_f1"] * 100 for n in ORDER]

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    fig.subplots_adjust(left=0.085, right=0.955, top=0.745, bottom=0.255)

    positions = np.arange(len(ORDER))
    width = 0.36
    colours = [ps.AMBER, ps.SLATE, ps.TEAL, ps.BLUE]
    ax.bar(positions - width / 2, accuracy, width, color=colours, label="Top-1 accuracy")
    ax.bar(positions + width / 2, macro, width, color=colours, alpha=0.5,
           label="Macro F1")
    for index, value in enumerate(accuracy):
        ax.text(index - width / 2, value + 1.4, f"{value:.2f}", ha="center",
                fontsize=10.6, color=ps.INK, fontweight="600")

    ax.set_xticks(positions)
    ax.set_xticklabels([NICE[n] for n in ORDER], fontsize=10.2, color=ps.INK)
    ax.set_ylabel("Score on the 10,000 published test images", fontsize=11,
                  color=ps.MUTED)
    ax.set_ylim(0, 108)
    ax.legend(loc="upper left", frameon=False, fontsize=10.4, labelcolor=ps.MUTED)
    ps.clean(ax, grid_axis="y")

    ps.title_block(
        fig, "The same 13,500 images and the same twelve epochs",
        "CIFAR-10, ten classes. Every model here was fitted on the identical "
        "training subset and scored once\non the full published test split.")

    cnn = models["custom_cnn"]["test"]["top1_accuracy"] * 100
    pre = models["pretrained_resnet18"]["test"]["top1_accuracy"] * 100
    ps.footnote(fig, [
        f"The custom network reaches {cnn:.2f}, which reproduces the "
        f"{data['reference_run']['test_accuracy']:.2f} this repository originally "
        f"reported. Starting from ImageNet weights instead of",
        f"random ones, on the same data and the same budget, reaches {pre:.2f}. "
        f"That is the comparison the original README was missing.",
    ])
    ps.save(fig, FIGURES, "01_model_comparison")


def figure_02_per_class(data: dict) -> None:
    """Which classes does the pretrained backbone actually help?"""
    classes = data["data"]["classes"]
    cnn = data["models"]["custom_cnn"]["test"]["per_class"]
    pre = data["models"]["pretrained_resnet18"]["test"]["per_class"]
    delta = [(pre[c]["f1"] - cnn[c]["f1"]) * 100 for c in classes]
    order = np.argsort(delta)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 6.6),
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.085, right=0.965, top=0.685, bottom=0.235, wspace=0.30)

    positions = np.arange(len(classes))
    width = 0.38
    ordered = [classes[i] for i in order]
    left.barh(positions - width / 2, [cnn[c]["f1"] * 100 for c in ordered], width,
              color=ps.TEAL, label="Custom CNN")
    left.barh(positions + width / 2, [pre[c]["f1"] * 100 for c in ordered], width,
              color=ps.BLUE, label="ResNet-18, pretrained")
    left.set_yticks(positions)
    left.set_yticklabels(ordered, fontsize=10.2, color=ps.INK)
    left.set_xlabel("F1 on the test split", fontsize=10.8, color=ps.MUTED)
    left.set_xlim(0, 105)
    # Above the axes: every row of bars starts at zero, so there is no corner
    # inside the panel that a legend can occupy without covering data.
    left.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2,
                frameon=False, fontsize=10, labelcolor=ps.MUTED)
    ps.clean(left, grid_axis="x")

    vehicles = {"airplane", "automobile", "ship", "truck"}
    right.barh(positions, [delta[i] for i in order],
               color=[ps.SLATE if ordered[i] in vehicles else ps.GREEN
                      for i in range(len(ordered))], height=0.62)
    for index, value in enumerate([delta[i] for i in order]):
        right.text(value + 0.8, index, f"{value:+.1f}", va="center", fontsize=9.8,
                   color=ps.INK)
    right.set_yticks(positions)
    right.set_yticklabels([])
    right.set_xlabel("F1 points gained from pretraining", fontsize=10.8,
                     color=ps.MUTED)
    right.set_xlim(0, max(delta) * 1.2)
    ps.clean(right, grid_axis="x")

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                          markerfacecolor=ps.GREEN, markeredgecolor=ps.GREEN,
                          label="animal"),
               plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                          markerfacecolor=ps.SLATE, markeredgecolor=ps.SLATE,
                          label="vehicle")]
    right.legend(handles=handles, loc="lower right", frameon=False, fontsize=10,
                 labelcolor=ps.MUTED)

    ps.title_block(
        fig, "Pretraining helps most on the animals",
        "Per-class F1 for both networks, and the gap between them, sorted by how "
        "much pretraining buys.")

    ps.footnote(fig, [
        "The four vehicle classes gain the least and the six animal classes the "
        "most, with bird and cat gaining more than forty F1 points",
        "each. ImageNet contains many fine-grained animal categories and "
        "comparatively few vehicles, which is a plausible reading, though",
        "nothing here isolates that as the cause.",
    ], y=0.105)
    ps.save(fig, FIGURES, "02_per_class")


def figure_03_confusion(data: dict) -> None:
    """What does each network confuse?"""
    classes = data["data"]["classes"]
    matrices = {
        "Custom CNN from scratch":
            np.array(data["models"]["custom_cnn"]["test"]["confusion_matrix"]),
        "ResNet-18, ImageNet weights":
            np.array(data["models"]["pretrained_resnet18"]["test"]["confusion_matrix"]),
    }
    ceiling = max(int((m - np.diag(np.diag(m))).max()) for m in matrices.values())

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.8))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.70, bottom=0.245, wspace=0.20)

    for axis, (title, matrix) in zip(axes, matrices.items(), strict=True):
        shown = matrix - np.diag(np.diag(matrix))
        axis.imshow(shown, cmap="Blues", vmin=0, vmax=ceiling)
        for row in range(len(classes)):
            for column in range(len(classes)):
                value = int(matrix[row][column])
                if row == column:
                    colour, weight = ps.FAINT, "400"
                elif shown[row][column] > ceiling * 0.55:
                    colour, weight = "white", "600"
                elif value == 0:
                    colour, weight = ps.HAIR, "400"
                else:
                    colour, weight = ps.INK, "400"
                axis.text(column, row, f"{value}", ha="center", va="center",
                          fontsize=7.6, color=colour, fontweight=weight)
        axis.set_xticks(range(len(classes)), classes, fontsize=8.4, rotation=45,
                        ha="right")
        axis.set_yticks(range(len(classes)), classes, fontsize=8.4)
        axis.set_xlabel("predicted", fontsize=10.2, color=ps.MUTED)
        axis.set_title(title, fontsize=12, color=ps.INK, pad=10, loc="left")
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)
    axes[0].set_ylabel("true", fontsize=10.2, color=ps.MUTED)

    worst = data["models"]["pretrained_resnet18"]["confusion_pairs"][0]
    ps.title_block(
        fig, "Both networks confuse cats with dogs first",
        "Counts on the 10,000 published test images, same class order and same "
        "colour scale. Correct counts on\nthe diagonal are greyed, because "
        "colouring them would flatten every error to white.")

    ps.footnote(fig, [
        f"{worst['true'].capitalize()} read as {worst['predicted']} is the largest "
        f"error cell for both, and it is the one pretraining shrinks least in "
        f"relative terms:",
        "it accounts for a larger share of the stronger model's remaining mistakes "
        "than of the weaker model's.",
    ])
    ps.save(fig, FIGURES, "03_confusion")


def figure_04_failures(data: dict) -> bool:
    """What do the mistakes look like?

    The only figure needing the images themselves. The other three read the
    recorded results alone, which is why continuous integration can regenerate
    them without a 170 MB download. This one says so and skips rather than
    failing the run.
    """
    from torchvision import datasets

    if not (ROOT / "data" / "cifar-10-batches-py").is_dir():
        print("  skipping 04_failure_examples: CIFAR-10 is not downloaded, and "
              "this is the only figure that needs the images")
        return False

    test = datasets.CIFAR10(root=str(ROOT / "data"), train=False, download=False)
    images = test.data
    classes = data["data"]["classes"]

    entry = data["models"]["pretrained_resnet18"]
    actual = np.array(entry["predictions"]["test_true"])
    predicted = np.array(entry["predictions"]["test_predicted"])

    # Chosen by rule: the four largest confusion cells, then the first four test
    # images falling in each, in index order. No image was picked for looking
    # convincing, and none was skipped for looking unconvincing.
    pairs = entry["confusion_pairs"][:4]
    columns = 4
    fig, axes = plt.subplots(len(pairs), columns, figsize=(10.2, 10.6))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.79, bottom=0.115,
                        hspace=0.36, wspace=0.06)

    for row, pair in enumerate(pairs):
        true_index = classes.index(pair["true"])
        predicted_index = classes.index(pair["predicted"])
        matching = np.flatnonzero((actual == true_index)
                                  & (predicted == predicted_index))[:columns]
        for column in range(columns):
            axis = axes[row][column]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(ps.HAIR)
            if column < len(matching):
                axis.imshow(images[matching[column]])
            else:
                axis.axis("off")
        axes[row][0].set_ylabel(
            f"{pair['true']}\nread as\n{pair['predicted']}\n({pair['count']})",
            fontsize=10, color=ps.INK, rotation=0, ha="right", va="center",
            labelpad=42)

    ps.title_block(
        fig, "The mistakes the stronger network still makes",
        "ResNet-18 with ImageNet weights. Rows are the four largest confusion "
        "cells; within each row the first\nfour test images in index order, so "
        "nothing was chosen for how it looks.", y=0.965)

    ps.footnote(fig, [
        "At 32 by 32 pixels a curled cat and a small dog carry little that "
        "separates them, and several of these would be hard to",
        "call from the image alone. That is the part of the remaining error a "
        "larger model of the same kind is least likely to fix.",
    ], y=0.075)
    ps.save(fig, FIGURES, "04_failure_examples")
    return True


def main() -> int:
    ps.apply()
    data = load()
    print(f"  reading {RESULTS.relative_to(ROOT).as_posix()}")
    figure_01_model_comparison(data)
    figure_02_per_class(data)
    figure_03_confusion(data)
    drawn = figure_04_failures(data)
    if not drawn and not (FIGURES / "04_failure_examples.png").is_file():
        raise SystemExit("  04_failure_examples.png is neither committed nor "
                         "reproducible here; download CIFAR-10 and rerun")
    print(f"  figures in {FIGURES.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
