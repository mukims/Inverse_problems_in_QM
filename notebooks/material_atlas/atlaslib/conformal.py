"""Relative split-conformal intervals: calibrate on held-out seeds, then cover (1 - alpha)."""
import numpy as np


def fit_relative(pred, true, alpha=0.1):
    scores = np.abs(np.asarray(true) - np.asarray(pred)) / np.maximum(np.asarray(pred), 1e-6)
    if scores.size == 0:
        raise ValueError("calibration set cannot be empty")
    n = scores.size
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def intervals(pred, q):
    pred = np.asarray(pred)
    return pred * (1 - q), pred * (1 + q)


def coverage(lo, hi, y):
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))
