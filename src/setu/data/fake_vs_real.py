"""Fake-vs-real classifier check (roadmap v4 data plan: "S1 vs real HKHPL
crops: target 70-80% accuracy -- not near-100%, which would mean the
synthetic images are too clean").

Deliberately a hand-crafted-feature logistic regression, not a trained
CNN or scikit-learn model: this is a cheap sanity check, not a reported
system component, and adding scikit-learn would be a new dependency
(CLAUDE.md: "Ask before ... adding a dependency"). numpy is already a
project dependency, so logistic regression is implemented directly on top
of it (gradient descent, no autograd needed for 4 features).

Features per patch, same family as setu.data.survey's page-level heuristics
(sharpness/contrast/brightness) plus ink_density, computed on small,
same-size crops so the two classes are compared on a level footing rather
than on page-vs-line-image resolution/scale differences that would make
this a trivial (and meaningless) classification task.

Usage:
    python -m setu.data.fake_vs_real --s1-dir data/s1 --real-dir data/real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from setu.render.textures import sample_patch
from setu.runlog import finish_run, start_run

PATCH_H, PATCH_W = 64, 256
SEED = 0


def _laplacian_variance(gray: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray.astype(np.float32)
    lap = (
        k[0, 1] * np.roll(g, 1, axis=0) + k[1, 0] * np.roll(g, 1, axis=1)
        + k[1, 1] * g + k[1, 2] * np.roll(g, -1, axis=1) + k[2, 1] * np.roll(g, -1, axis=0)
    )
    return float(np.var(lap[1:-1, 1:-1]))


def extract_features(patch: np.ndarray) -> np.ndarray:
    """patch: (H, W) uint8, ink-dark-on-light (normal viewing convention,
    matching both the saved S1 PNGs and the real HKHPL photos as-is)."""
    g = patch.astype(np.float32)
    sharpness = _laplacian_variance(g)
    contrast = float(g.std())
    brightness = float(g.mean())
    ink_density = float((g < 128).mean())
    return np.array([sharpness, contrast, brightness, ink_density], dtype=np.float64)


def _s1_patch(image_path: Path, rng: np.random.Generator) -> np.ndarray:
    img = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)  # ink-dark-on-light, as saved
    h, w = img.shape
    if h >= PATCH_H:
        y0 = int(rng.integers(0, h - PATCH_H + 1))
        img = img[y0 : y0 + PATCH_H]
    else:
        img = np.pad(img, ((0, PATCH_H - h), (0, 0)), constant_values=255)  # pad with "page", not "ink"
    if w >= PATCH_W:
        x0 = int(rng.integers(0, w - PATCH_W + 1))
        img = img[:, x0 : x0 + PATCH_W]
    else:
        img = np.pad(img, ((0, 0), (0, PATCH_W - w)), constant_values=255)
    return img


def _load_s1_image_paths(s1_dir: Path) -> list[Path]:
    manifest = s1_dir / "manifest.jsonl"
    paths = []
    for line in manifest.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("skipped"):
            continue
        paths.append(s1_dir / rec["image_path"].replace("\\", "/"))  # see train.py: Windows-written manifest
    return paths


def train_logistic_regression(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 2000, seed: int = SEED
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / n
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    p = 1.0 / (1.0 + np.exp(-z))
    return (p >= 0.5).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-dir", type=Path, default=Path("data/s1"))
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--n-per-class", type=int, default=1500)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)  # CLAUDE.md rule 6

    s1_paths = _load_s1_image_paths(args.s1_dir)
    if not s1_paths:
        raise RuntimeError(f"No S1 images found under {args.s1_dir} -- render S1 first.")
    n_s1 = min(args.n_per_class, len(s1_paths))
    s1_sample_idx = rng.choice(len(s1_paths), size=n_s1, replace=False)

    features, labels = [], []
    for idx in s1_sample_idx:
        patch = _s1_patch(s1_paths[idx], rng)
        features.append(extract_features(patch))
        labels.append(0)  # 0 = fake/synthetic

    for _ in range(args.n_per_class):
        patch, is_real = sample_patch(args.real_dir, (PATCH_H, PATCH_W), rng)
        if not is_real:
            raise RuntimeError(
                f"sample_patch fell back to the procedural placeholder -- no real images found "
                f"under {args.real_dir}. This check is meaningless without real HKHPL photos."
            )
        features.append(extract_features(patch))
        labels.append(1)  # 1 = real

    X = np.stack(features)
    y = np.array(labels, dtype=np.float64)

    # standardize features (fit on train only, applied to both splits)
    perm = rng.permutation(len(y))
    n_test = int(len(y) * args.test_fraction)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    mean, std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    w, b = train_logistic_regression(X_norm[train_idx], y[train_idx])

    train_pred = predict(X_norm[train_idx], w, b)
    test_pred = predict(X_norm[test_idx], w, b)
    train_acc = float((train_pred == y[train_idx]).mean())
    test_acc = float((test_pred == y[test_idx]).mean())

    feature_names = ["sharpness", "contrast", "brightness", "ink_density"]

    run_dir = start_run(
        "fake_vs_real_classifier",
        {
            "seed": SEED,
            "s1_dir": str(args.s1_dir),
            "real_dir": str(args.real_dir),
            "n_per_class": args.n_per_class,
            "patch_size": [PATCH_H, PATCH_W],
            "test_fraction": args.test_fraction,
            "features": feature_names,
            "method": "hand-rolled logistic regression (numpy gradient descent) -- no new "
                      "dependency added for what is a cheap sanity check, not a reported system",
        },
    )
    results = {
        "n_total": len(y),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "target_range": [0.70, 0.80],
        "in_target_range": 0.70 <= test_acc <= 0.80,
        "learned_weights": dict(zip(feature_names, w.tolist())),
        "bias": b,
        "feature_means_train": dict(zip(feature_names, mean.tolist())),
        "feature_stds_train": dict(zip(feature_names, std.tolist())),
    }
    finish_run(run_dir, results)

    print(f"S1 (fake) patches: {n_s1}, real HKHPL patches: {args.n_per_class}")
    print(f"Train accuracy: {train_acc:.3f}  Test accuracy: {test_acc:.3f}")
    if results["in_target_range"]:
        print("Test accuracy is in the roadmap's 70-80% target band.")
    elif test_acc > 0.80:
        print(
            f"Test accuracy {test_acc:.3f} is ABOVE the 70-80% target band -- synthetic images "
            f"may be too easy to distinguish from real ones (too clean, or a feature is trivially "
            f"separating -- check learned_weights in results.json for which one dominates)."
        )
    else:
        print(f"Test accuracy {test_acc:.3f} is below the 70-80% target band.")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
