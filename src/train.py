"""Training loop."""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.model import ShapeClassifier
from src.dataset import ShapeDataModule


def train(config):
    """Train the shape classifier and return trainer, model, and datamodule."""
    datamodule = ShapeDataModule(config)
    model = ShapeClassifier(config)

    checkpoint_dir = config["output"]["checkpoint_dir"]
    log_dir = config["output"]["log_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=log_dir,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, datamodule=datamodule)

    return trainer, model, datamodule
