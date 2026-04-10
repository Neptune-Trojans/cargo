import argparse
import os
import sys

import yaml

from src.train import train
from src.evaluate import evaluate
from scripts.generate_synthetic_lines import main as generate_lines
from scripts.generate_synthetic_parabolas import main as generate_parabolas
from scripts.generate_synthetic_sines import main as generate_sines


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_real_data(classes):
    """Check that real data exists for all classes. Exit if missing."""
    missing = []
    for cls in classes:
        class_dir = os.path.join("data", "real_data", cls)
        if not os.path.isdir(class_dir) or not os.listdir(class_dir):
            missing.append(class_dir)

    if missing:
        print("ERROR: Real data is missing in the following folders:")
        for path in missing:
            print(f"  - {path}")
        print("Please copy real data images before running training.")
        sys.exit(1)

    print("Real data validated for all classes.")


def ensure_synthetic_data(classes):
    """Check that synthetic data exists for all classes. Regenerate all if any is missing."""
    missing = False
    for cls in classes:
        class_dir = os.path.join("data", "synthetic", cls)
        if not os.path.isdir(class_dir) or not os.listdir(class_dir):
            missing = True
            break

    if missing:
        print("Synthetic data missing, regenerating all classes...")
        generate_lines()
        generate_parabolas()
        generate_sines()

    print("Synthetic data validated for all classes.")


def main():
    parser = argparse.ArgumentParser(description="Cargo - Deep Learning Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    classes = config["data"]["classes"]

    validate_real_data(classes)
    ensure_synthetic_data(classes)

    print(f"Starting training with model: {config['model']['name']}")
    print(f"Classes: {classes}")

    trainer, model, datamodule = train(config)
    evaluate(trainer, model, datamodule, config)


if __name__ == "__main__":
    main()
