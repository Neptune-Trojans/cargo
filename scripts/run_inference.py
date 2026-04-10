"""Example script: load a trained model and classify a real image."""

import argparse
import time

import yaml

from src.inference import load_model, load_thresholds, classify_image


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--config", type=str, default="../configs/default.yaml")
    parser.add_argument("--model", type=str, default="../outputs/model.pt")
    parser.add_argument("--thresholds", type=str, default="../outputs/thresholds.json")
    parser.add_argument("--image", type=str, default="../data/real_data/line/line_00000.png")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = load_model(config, args.model)
    thresholds = load_thresholds(args.thresholds)

    start = time.perf_counter()
    prediction, probs = classify_image(model, args.image, config, thresholds)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"Classify time: {elapsed_ms:.2f} ms")

    classes = config["data"]["classes"]
    print(f"\nImage: {args.image}")
    print(f"Prediction: {prediction.value}")
    print("Probabilities:")
    for cls, prob in zip(classes, probs):
        print(f"  {cls}: {prob:.4f}")


if __name__ == "__main__":
    main()
