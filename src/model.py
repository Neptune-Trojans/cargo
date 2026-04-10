"""Model architecture definitions."""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class ShapeClassifier(pl.LightningModule):
    """MobileNetV3-Small classifier for line/parabola/sine shapes."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.classes = config["data"]["classes"]
        num_classes = config["model"]["num_classes"]
        dropout = config["model"].get("dropout", 0.5)

        weights = "IMAGENET1K_V1" if config["model"]["pretrained"] else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        self.test_preds.append(preds)
        self.test_targets.append(y)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).cpu().numpy()
        targets = torch.cat(self.test_targets).cpu().numpy()
        self.test_preds.clear()
        self.test_targets.clear()

        # Classification report
        report = classification_report(targets, preds, target_names=self.classes)
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        print("Confusion Matrix:")
        print(cm)

        # Save confusion matrix plot
        output_dir = self.config["output"]["checkpoint_dir"].rsplit("/", 1)[0]
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(len(self.classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.classes)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "syn_confusion_matrix.png")
        fig.savefig(cm_path)
        plt.close(fig)
        print(f"\nConfusion matrix saved to {cm_path}")

        # Save test results to file
        report_path = os.path.join(output_dir, "syn_test_results.txt")
        with open(report_path, "w") as f:
            f.write("TEST RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
        print(f"Test results saved to {report_path}")

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        wd = self.config["training"]["weight_decay"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["training"]["epochs"]
        )
        return [optimizer], [scheduler]
