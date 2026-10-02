# tests/meaning/test_split.py
import numpy as np
import pytest

from meaning.split import check_same_ribbons, seed_split


def test_seed_split_boundaries():
    tr, va, te = seed_split(np.array([0, 699, 700, 849, 850, 999]))
    assert tr.tolist() == [True, True, False, False, False, False]
    assert va.tolist() == [False, False, True, True, False, False]
    assert te.tolist() == [False, False, False, False, True, True]


def test_check_same_ribbons():
    check_same_ribbons(["a", "b"], ["b", "a"])
    with pytest.raises(ValueError, match="only in second"):
        check_same_ribbons(["a"], ["a", "b"])
