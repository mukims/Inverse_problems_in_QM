# tests/atlas/test_multistore.py
import numpy as np
import pytest
from atlaslib import MultiStore
from atlaslib.store import CloudStore

E = np.arange(5) * 0.1


def _put(root, mid, level):
    st = CloudStore(root)
    st.write_pristine(mid, E, np.full(5, level))
    st.write_cloud(mid, 0.01, 1, np.full((2, 5), level - 0.5), np.array([0, 1]), E)
    return st


def test_reads_each_model_from_its_own_store(tmp_path):
    a = _put(tmp_path / "a", "x/armchair/N7", 3.0)
    b = _put(tmp_path / "b", "y/zigzag/N9", 5.0)
    ms = MultiStore(a, b)
    assert ms.models() == ["x/armchair/N7", "y/zigzag/N9"]
    assert ms.read_pristine("y/zigzag/N9")[1][0] == 5.0
    assert ms.densities("x/armchair/N7") == [0.01]
    assert ms.read_cloud("x/armchair/N7", 0.01)[0][0, 0] == 2.5


def test_accepts_paths(tmp_path):
    _put(tmp_path / "a", "x/armchair/N7", 3.0)
    assert MultiStore(tmp_path / "a").models() == ["x/armchair/N7"]


def test_model_in_two_stores_is_refused(tmp_path):
    a = _put(tmp_path / "a", "x/armchair/N7", 3.0)
    b = _put(tmp_path / "b", "x/armchair/N7", 3.0)
    with pytest.raises(ValueError, match="more than one store"):
        MultiStore(a, b)


def test_unknown_model_raises_key_error(tmp_path):
    ms = MultiStore(_put(tmp_path / "a", "x/armchair/N7", 3.0))
    with pytest.raises(KeyError, match="none of the stores"):
        ms.read_pristine("z/armchair/N5")
