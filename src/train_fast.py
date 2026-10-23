"""
Fast training script for CNN on CIFAR-10. This is the script that produced the
originally published result in results/test_results.json.

It draws a seeded random subset of 15,000 of the 50,000 CIFAR-10 training images
(see subset_size below), splits it 90/10, trains for 12 epochs and scores on the
full published 10,000-image test set. Sized for a modest GPU such as an MX330.

src/train.py is the one that uses all 50,000 images, over 50 epochs. It has never
been run, so no result in this repository comes from it.

For the fully seeded harness that reproduced this run and added the comparison
arms either side of it, see src/benchmark.py.
"""

import json
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    # python -m src.train_fast, from the repository root, which is the form the
    # README uses for every other script here.
    from src.model import CIFAR10CNN
except ImportError:
    # python src/train_fast.py, which puts src/ on sys.path instead.
    from model import CIFAR10CNN

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def train_and_evaluate():
    """Full training and evaluation pipeline."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    sys.stdout.flush()

    MAX_EPOCHS = 12
    BATCH_SIZE = 128
    LR = 0.002

    # ============================================================
    # DATA
    # ============================================================
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Use a subset of 15000 samples for efficient training on modest GPU
    subset_size = 15000
    rng_gen = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(len(full_train), generator=rng_gen)[
        :subset_size
    ].tolist()
    subset_train = torch.utils.data.Subset(full_train, all_indices)

    # Train/val split from the subset
    train_size = int(0.9 * subset_size)
    val_size = subset_size - train_size
    train_set, val_set_sub = random_split(
        subset_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_set_clean = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_transform
    )
    val_original_indices = [all_indices[i] for i in val_set_sub.indices]
    val_set_final = torch.utils.data.Subset(val_set_clean, val_original_indices)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_set_final,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"Train: {len(train_set)}, Val: {val_size}, Test: {len(test_set)}")
    sys.stdout.flush()

    # ============================================================
    # MODEL
    # ============================================================
    model = CIFAR10CNN().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR, epochs=MAX_EPOCHS, steps_per_epoch=len(train_loader)
    )

    # ============================================================
    # TRAINING
    # ============================================================
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= total
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_acc = 100.0 * correct / total

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:2d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.0f}s"
        )
        sys.stdout.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    sys.stdout.flush()

    # ============================================================
    # TEST EVALUATION
    # ============================================================
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    )
    model.eval()

    all_preds, all_labels = [], []
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= total
    test_acc = 100.0 * correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    print(f"\n{report}")
    sys.stdout.flush()

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)

    # ============================================================
    # VISUALIZATIONS
    # ============================================================

    # 1. Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train", color="#2196F3")
    ax1.plot(epochs, history["val_loss"], label="Validation", color="#F44336")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", color="#2196F3")
    ax2.plot(epochs, history["val_acc"], label="Validation", color="#F44336")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "01_training_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved: 01_training_curves.png")
    sys.stdout.flush()

    # 2. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    # Drawn with matplotlib. This was the only call needing seaborn, and it was
    # imported at module scope, so the file could not be imported without it.
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, shrink=0.82)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    midpoint = cm.max() / 2.0
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            ax.text(column, row, f"{cm[row, column]:d}", ha="center", va="center",
                    color="white" if cm[row, column] > midpoint else "black",
                    fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix \u2014 Test Accuracy: {test_acc:.2f}%")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved: 02_confusion_matrix.png")
    sys.stdout.flush()

    # 3. Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    bars = ax.bar(CLASSES, per_class_acc, color=colors, edgecolor="white")
    for bar, acc in zip(bars, per_class_acc, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Test Accuracy")
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "03_per_class_accuracy.png"), dpi=150)
    plt.close(fig)
    print("Saved: 03_per_class_accuracy.png")
    sys.stdout.flush()

    # 4. Sample predictions
    test_set_viz = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transforms.ToTensor()
    )
    fig, axes = plt.subplots(4, 8, figsize=(16, 9))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(test_set_viz), 32, replace=False)

    for idx_i, sample_idx in enumerate(indices):
        ax = axes[idx_i // 8, idx_i % 8]
        image, label = test_set_viz[sample_idx]
        img_np = image.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        pred = all_preds[sample_idx]
        is_correct = pred == label
        ax.imshow(img_np)
        color = "green" if is_correct else "red"
        ax.set_title(f"{CLASSES[pred]}", fontsize=8, color=color, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Sample Predictions (Green=Correct, Red=Wrong)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_sample_predictions.png"), dpi=150)
    plt.close(fig)
    print("Saved: 04_sample_predictions.png")
    sys.stdout.flush()

    # 5. Learning rate schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    total_steps = list(range(1, len(history["lr"]) + 1))
    ax.plot(total_steps, history["lr"], color="#FF9800", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("OneCycleLR Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "05_learning_rate.png"), dpi=150)
    plt.close(fig)
    print("Saved: 05_learning_rate.png")
    sys.stdout.flush()

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results = {
        "test_accuracy": round(test_acc, 2),
        "test_loss": round(test_loss, 4),
        "best_val_accuracy": round(best_val_acc, 2),
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"]),
        "training_time_s": round(total_time, 1),
        "n_parameters": model.count_parameters(),
        "per_class_accuracy": {
            CLASSES[i]: round(float(per_class_acc[i]), 1) for i in range(10)
        },
    }

    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    hparams = {
        "architecture": "Custom CNN (3 conv blocks + FC)",
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "weight_decay": 1e-4,
        "dropout": 0.25,
        "augmentation": [
            "RandomCrop(32, pad=4)",
            "RandomHorizontalFlip",
            "ColorJitter",
        ],
    }
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    print(f"\nAll results saved to {output_dir}")
    print(f"Final: Test Acc={test_acc:.2f}%, Params={model.count_parameters():,}")
    sys.stdout.flush()
    return results


if __name__ == "__main__":
    train_and_evaluate()
