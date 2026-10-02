# notebooks/material_atlas/meaning/split.py
"""Configuration-seed split (LOGBOOK Bug #7) and a guard that compared encoders saw the same ribbons."""
import numpy as np

TRAIN_MAX, VAL_MAX = 699, 849


def seed_split(seeds):
    s = np.asarray(seeds)
    return s <= TRAIN_MAX, (s > TRAIN_MAX) & (s <= VAL_MAX), s > VAL_MAX


def check_same_ribbons(a, b):
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        raise ValueError(f"ribbon sets differ: only in first {only_a}, only in second {only_b}")
