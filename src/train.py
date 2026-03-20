"""
Training and evaluation pipeline for CNN on CIFAR-10.

Includes:
- Data augmentation (random crop, horizontal flip, color jitter)
- Training with learning rate scheduling
- Early stopping
- Per-class accuracy analysis
- Confusion matrix visualization
- Sample predictions grid
"""

import os
import json
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report

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


def get_dataloaders(data_dir: str, batch_size: int = 128):
    """Create train/val/test dataloaders with augmentation."""

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

    # Split train into train+val (use full dataset for proper training)
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Val set uses test transform (no augmentation)
    val_set_clean = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_transform
    )
    val_indices = val_set.indices
    val_set_final = torch.utils.data.Subset(val_set_clean, val_indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_set_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


def train_and_evaluate(
    data_dir: str = "../data",
    output_dir: str = "../results",
    max_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = None,
):
    """Full training and evaluation pipeline."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)

    # Model
    model = CIFAR10CNN().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
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

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
                f"LR: {current_lr:.2e} | Time: {elapsed:.0f}s"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}. Best: epoch {best_epoch} ({best_val_acc:.1f}%)"
            )
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    # Load best model for evaluation
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    )
    model.eval()

    # ============================================================
    # TEST SET EVALUATION
    # ============================================================
    all_preds, all_labels, all_probs = [], [], []
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss /= total
    test_acc = 100.0 * correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Per-class accuracy
    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    print(f"\n{report}")

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

    # 2. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — Test Accuracy: {test_acc:.2f}%")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved: 02_confusion_matrix.png")

    # 3. Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    bars = ax.bar(CLASSES, per_class_acc, color=colors, edgecolor="white")
    for bar, acc in zip(bars, per_class_acc):
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

    # 4. Sample predictions
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    test_set_viz = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
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
        correct = pred == label

        ax.imshow(img_np)
        color = "green" if correct else "red"
        ax.set_title(f"{CLASSES[pred]}", fontsize=8, color=color, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Sample Predictions (Green=Correct, Red=Wrong)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_sample_predictions.png"), dpi=150)
    plt.close(fig)
    print("Saved: 04_sample_predictions.png")

    # 5. Learning rate schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["lr"], color="#FF9800", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Annealing Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "05_learning_rate.png"), dpi=150)
    plt.close(fig)
    print("Saved: 05_learning_rate.png")

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
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
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
    return results


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    train_and_evaluate(data_dir=data_dir, output_dir=results_dir)
