"""Evaluation and metrics."""

import argparse
import torch
import os
import yaml

from src.calibrate import calibrate
from src.inference import load_model


def evaluate(trainer, model, datamodule, config):
    """Run test evaluation and save the final model."""
    # Test with best checkpoint
    trainer.test(model, datamodule=datamodule)

    # Save final model
    output_dir = os.path.dirname(config["output"]["checkpoint_dir"])
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Calibrate per-class confidence thresholds on real data
    calibrate(model, config)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on real data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="outputs/model.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Loading model from {args.model}")
    model = load_model(config, args.model)
    calibrate(model, config)


if __name__ == "__main__":
    main()