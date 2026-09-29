import json
from pathlib import Path
import numpy as np
import pytest
from atlaslib import CloudStore, InputSpec, Registry, RibbonModel
from notebooks.material_atlas.build_atlas_v2 import HELD_OUT, TRAIN


def test_held_out_partition():
    for key, widths in HELD_OUT.items():
        train_pool = list(TRAIN[key])
        for w in widths:
            assert w in train_pool, f"Held-out width {w} not in train set for {key}"
        remaining = [w for w in train_pool if w not in widths]
        assert len(remaining) > 0, f"No training widths remaining for {key}"
