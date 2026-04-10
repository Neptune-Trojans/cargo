"""Shared inference utilities: model loading, preprocessing, and prediction."""

import json
import time
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import ShapeClassifier


class ShapeClass(Enum):
    LINE = "line"
    PARABOLA = "parabola"
    SINE = "sine"
    UNKNOWN = "unknown"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform(image_size):
    """Return the standard evaluation transform."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(image_size):
    """Return the training transform with augmentation.

    ColorJitter and GaussianBlur bridge the domain gap between
    synthetic data (pure black dots, sharp edges) and real data
    (dark gray dots, softer edges).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(config, model_path):
    """Load a trained model from a saved state dict."""
    model = ShapeClassifier(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def predict_image(model, image_path, transform):
    """Run inference on a single image, return class probabilities."""
    img = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def load_thresholds(thresholds_path):
    """Load per-class confidence thresholds from a JSON file."""
    with open(thresholds_path) as f:
        return json.load(f)


def classify_image(model, image_path, config, thresholds=None):
    """Classify a single image, applying per-class confidence thresholds.

    Returns ShapeClass.UNKNOWN if no class exceeds its threshold.
    """
    classes = config["data"]["classes"]
    transform = get_eval_transform(config["data"]["image_size"])

    probs = predict_image(model, image_path, transform)

    if thresholds is None:
        return ShapeClass.UNKNOWN, probs

    best_class = None
    best_prob = -1.0
    for idx, class_name in enumerate(classes):
        if probs[idx] >= thresholds[class_name] and probs[idx] > best_prob:
            best_prob = probs[idx]
            best_class = class_name

    if best_class is None:
        return ShapeClass.UNKNOWN, probs

    return ShapeClass(best_class), probs