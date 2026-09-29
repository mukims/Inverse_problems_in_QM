import numpy as np
from atlaslib import Atlas, InputSpec, Registry
try:
    from toy import E, toy_spectrum, toy_store
except ImportError:
    from tests.atlas.toy import E, toy_spectrum, toy_store

FAST = dict(latent=8, epochs=6, patience=3, k=7, refs_per_model=200, threads=2)


def test_add_models_keeps_old_answers_and_reports_familiarity(tmp_path):
    store, models = toy_store(tmp_path, materials=(("alpha", 1.0),))
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    probe = toy_spectrum(1.0, 9, 0.01, 90_000)[None]
    before = atlas.locate(probe, E, 3.0)
    n_refs = len(atlas.refs)

    store2, new = toy_store(tmp_path / "b", materials=(("gamma", 6.0),), widths=(9,))
    for m in new:
        reg.add(m)
        e_t, p = store2.read_pristine(m.model_id); store.write_pristine(m.model_id, e_t, p)
        for d in store2.densities(m.model_id):
            c, s = store2.read_cloud(m.model_id, d); store.write_cloud(m.model_id, d, 1, c, s, e_t)
    report = atlas.add_models(store, reg, [m.model_id for m in new])

    assert set(report) == {"gamma/armchair/N9"} and 0.0 <= report["gamma/armchair/N9"] <= 1.0
    assert len(atlas.refs) > n_refs
    assert atlas.locate(probe, E, 3.0)[0].material == before[0].material
