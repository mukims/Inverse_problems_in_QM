import numpy as np
import pytest

from shazam.arrangement import misfit
from shazam.inverse import closeness as cl
from shazam.inverse import sim_reward as sr


def _brute(library, target, k):
    d = np.array([misfit(sr.smooth(row, 5), sr.smooth(target, 5)) for row in library])
    order = np.argsort(d, kind="stable")[:k]
    return order, d[order]


def test_nearest_spectra_match_a_brute_force_search_on_the_smoothed_spectra():
    rng = np.random.default_rng(0)
    library = rng.random((40, 416)).astype(np.float32)
    targets = (library[[3, 17, 29]] + 0.05 * rng.normal(size=(3, 416))).astype(np.float32)
    rows, d = cl.nearest_spectra(library, targets, k=4, chunk=2)
    assert rows.shape == d.shape == (3, 4) and list(rows[:, 0]) == [3, 17, 29]
    for i in range(3):
        want_rows, want_d = _brute(library, targets[i], 4)
        assert list(rows[i]) == list(want_rows)
        assert d[i] == pytest.approx(want_d, rel=1e-9)
        assert np.all(np.diff(d[i]) >= 0)                                  # nearest first


def test_nearest_spectra_never_return_the_excluded_row():
    rng = np.random.default_rng(1)
    library = rng.random((20, 416)).astype(np.float32)
    rows, d = cl.nearest_spectra(library, library[[4, 9]], k=3, exclude=[4, 9])
    assert 4 not in rows[0] and 9 not in rows[1] and np.all(d > 0)
    same, d0 = cl.nearest_spectra(library, library[[4, 9]], k=1)
    assert list(same[:, 0]) == [4, 9] and np.allclose(d0, 0)


def test_nearest_spectra_refuse_more_neighbours_than_the_library_holds():
    library = np.zeros((3, 416), np.float32)
    with pytest.raises(ValueError):
        cl.nearest_spectra(library, library[:1], k=3, exclude=[0])


def test_noise_scale_is_the_m0_model_row_by_row():
    m0 = sr.M0Model([-6.0, -0.9, 0.3])
    w, c = np.array([4, 9, 22]), np.array([0.0025, 0.01, 0.05])
    got = cl.noise_scale(m0, w, c)
    assert got.shape == (3,) and got == pytest.approx([m0(a, b) for a, b in zip(w, c)])


def test_summary_reports_the_first_and_the_closest_candidate_and_counts_missing_ones_as_misses():
    nan = np.nan
    ratio = np.array([[0.5, 3.0, nan], [4.0, 1.5, 8.0], [nan, nan, nan]])
    right = np.array([[True, False, False], [False, True, False], [False, False, False]])
    s = cl.summarise(ratio, right, groups=["H", "H", "CHK"])
    assert s["n_targets"] == 3 and s["candidates_per_target"] == pytest.approx(5 / 3)
    assert s["first"]["median"] == pytest.approx(2.25) and s["best"]["median"] == pytest.approx(1.0)
    assert s["all"]["median"] == pytest.approx(3.0)
    assert s["best"]["percentiles"] == pytest.approx(list(np.percentile([0.5, 1.5], cl.PERCENTILES)))
    assert s["best_within_1"] == pytest.approx(1 / 3) and s["best_within_2"] == pytest.approx(2 / 3)
    assert s["first_right_family"] == pytest.approx(1 / 3)
    assert s["any_right_family"] == pytest.approx(2 / 3)
    assert s["closest_is_right_family"] == pytest.approx(2 / 3)
    assert s["by_group"]["H"] == dict(n=2, first_median=pytest.approx(2.25), best_median=pytest.approx(1.0),
                                      best_within_2=pytest.approx(1.0))
    assert s["by_group"]["CHK"] == dict(n=1, first_median=None, best_median=None, best_within_2=0.0)


def test_summary_without_family_flags_leaves_them_out():
    s = cl.summarise(np.array([[1.0, 2.0]]))
    assert "first_right_family" not in s and "by_group" not in s and s["best"]["median"] == 1.0


# ---------------------------------------------------------------- the scoreboard script (scripts/evaluate/eval_closeness.py)

import importlib.util
import json
from pathlib import Path

from shazam.inverse import realism as rl
from shazam.inverse.data import EDGE_CODES, LEAD_CODES
from shazam.inverse.runio import RunLog
from tbribbon.parent import named_material
from tbribbon.parent_prior import FIELDS, Draw, in_prior, sample_prior

_SPEC = importlib.util.spec_from_file_location(
    "eval_closeness", Path(__file__).resolve().parents[2] / "scripts" / "evaluate" / "eval_closeness.py")
ec = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ec)


def _arrays(n, seed=0, start=0, known=()):
    """(arrays in the stored layout, their draws): n prior draws with random spectra, draw indices from `start`. The
    rows listed in `known` hold a known material instead (graphene and hBN in turn), the others never do."""
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < n:
        d = sample_prior(rng)
        if not rl.is_near_known(d):
            draws.append(d)
    for j, row in enumerate(known):
        draws[row] = Draw(named_material(("graphene-ideal", "hbn")[j % 2]), "armchair", 5 + j, 0.01)
    arrays = dict(index=np.arange(start, start + n), x=rng.random((n, 416)).astype(np.float32),
                  on=np.array([d.params.pattern for d in draws], np.int8),
                  lead=np.array([LEAD_CODES[d.params.lead] for d in draws], np.int8),
                  params=np.array([[getattr(d.params, f) for f in FIELDS] for d in draws]),
                  edge=np.array([EDGE_CODES[d.edge] for d in draws], np.int8),
                  width=np.array([d.width for d in draws], np.int16), density=np.array([d.density for d in draws]))
    return arrays, draws


def _model(seed):
    from test_inverse_finetune import _big_model
    return _big_model(seed).eval()


def test_candidate_seeds_are_unique_per_target_and_clear_of_the_other_seed_ranges():
    seeds = [s for p in range(300) for s in ec.candidate_seeds(p, ec.MAX_K)]
    assert len(set(seeds)) == len(seeds)
    assert min(seeds) > 3_500_000_000 + 128 and max(seeds) < 4 * 10**9       # above the pilot's seeds, below phase 2's
    assert ec.candidate_seeds(7, 2) == ec.candidate_seeds(7, 3)[:2]
    with pytest.raises(ValueError):
        ec.candidate_seeds(0, ec.MAX_K + 1)


def test_lookup_proposes_the_stored_hosts_of_the_nearest_train_spectra():
    train, draws = _arrays(30, seed=1, start=100)
    targets = {k: v[[5, 12]] for k, v in _arrays(30, seed=2)[0].items()}
    targets["x"] = train["x"][[7, 21]] + np.float32(0.01)
    propose = ec.lookup_proposer(train, targets, k=3)
    first = propose(0)
    assert len(first) == 3 and first[0]["draw"] == draws[7] and first[0]["source_index"] == 107
    assert first[0]["stored_misfit"] == pytest.approx(1e-4, rel=1e-4)
    assert first[0]["stored_misfit"] < first[1]["stored_misfit"] < first[2]["stored_misfit"]
    assert propose(1)[0]["draw"] == draws[21]


def test_lookup_of_known_materials_proposes_only_hosts_near_one():
    train, draws = _arrays(40, seed=1, start=100, known=(3, 8, 20, 31))
    targets = {k: v[[5, 12]] for k, v in _arrays(30, seed=2)[0].items()}
    targets["x"] = train["x"][[7, 20]] + np.float32(0.01)                  # row 7 is not a known material, row 20 is
    free, known = ec.lookup_proposer(train, targets, k=3), ec.lookup_proposer(train, targets, k=3, known_only=True)
    assert free(0)[0]["source_index"] == 107
    assert {c["source_index"] for c in known(0)} <= {103, 108, 120, 131} and len(known(0)) == 3
    assert known(1)[0]["source_index"] == 120 and known(1)[0]["draw"] == draws[20]
    assert all(rl.is_near_known(c["draw"]) for i in (0, 1) for c in known(i))
    with pytest.raises(ValueError):
        ec.lookup_proposer(train, targets, k=5, known_only=True)           # only 4 known-material rows


def test_host_record_keeps_what_realism_needs_and_reads_back_to_the_same_distance():
    d = Draw(named_material("hbn"), "zigzag", 11, 0.02)
    h = ec.host_record(d)
    assert h["pattern"] == "H" and h["edge"] == "zigzag" and h["density"] == 0.02
    assert h["ev"] == pytest.approx(list(rl.physical_ev(d))) and json.loads(json.dumps(h)) == h
    names, dist = ec.hosts_nearest_material([h, ec.host_record(sample_prior(np.random.default_rng(9)))])
    assert names[0] == "hbn" and dist[0] == 0.0 and dist[1] > 1


def test_summary_reports_the_hosts_near_a_known_material():
    import dataclasses
    from tbribbon.parent import HexParams
    graphene = named_material("graphene-ideal")
    near = ec.host_record(Draw(graphene, "armchair", 9, 0.01))
    far_h = ec.host_record(Draw(dataclasses.replace(graphene, s=4.0), "armchair", 9, 0.01))       # 4.0 eV: 4.8 tolerances
    far_ch = ec.host_record(Draw(HexParams(True, True, False, "H", 2.7, t_AB=1.0, t_CC=0.5, t_CH=0.3), "zigzag", 9, 0.01))
    cand = lambda ratio, host: dict(ratio=ratio, family_right=False, width=9, host=host)
    records = [dict(index=1, pattern="H", m0=1.0, candidates=[cand(2.0, far_h), cand(4.0, near)]),
               dict(index=2, pattern="CH", m0=1.0, candidates=[cand(3.0, far_ch)])]
    s = ec.summary(records, k=2)["near_known"]
    assert s["share_of_hosts"] == pytest.approx(1 / 3) and s["targets_with_one"] == 0.5
    assert s["best"]["median"] == 4.0 and s["best_within_2"] == 0.0 and s["first"]["n"] == 0
    wide = ec.summary(records, k=2, rel_tol=0.6, abs_tol=0.1)["near_known"]     # 4.0 eV is now near graphene; no C+H material
    assert wide["share_of_hosts"] == pytest.approx(2 / 3) and wide["targets_with_one"] == 0.5
    assert wide["best"]["median"] == 2.0 and wide["best_within_2"] == 0.5 and wide["first"]["n"] == 1
    for r in records:
        for c in r["candidates"]:
            del c["host"]
    assert "near_known" not in ec.summary(records, k=2)                     # records written before hosts were kept


def test_truth_proposes_the_targets_own_host_as_the_floor_of_the_scoreboard():
    targets, draws = _arrays(3, seed=6, start=80)
    propose = ec.truth_proposer(targets)
    assert [propose(i) for i in range(3)] == [[dict(draw=d)] for d in draws]


def test_model_proposals_are_seeded_by_the_draw_index_and_lie_inside_the_prior():
    targets, _ = _arrays(3, seed=3, start=40)
    propose = ec.model_proposer(_model(0), targets, k=3, n_samples=400)
    a, b = propose(1), propose(1)
    assert 1 <= len(a) <= 3 and [c["draw"] for c in a] == [c["draw"] for c in b]
    assert all(in_prior(c["draw"]) for c in a)
    p = [c["probability"] for c in a]
    assert p == sorted(p, reverse=True) and 0 < sum(p) <= 1 + 1e-9
    assert [c["draw"] for c in propose(0)] != [c["draw"] for c in a]            # another target, another answer
    s = ec.model_proposer(_model(0), targets, k=3, n_samples=400, mode="samples")(1)
    assert len(s) == 3 and all(in_prior(c["draw"]) for c in s)
    with pytest.raises(ValueError):
        ec.model_proposer(_model(0), targets, k=3, n_samples=400, mode="best")


def _scored(tmp_path, calls):
    targets, draws = _arrays(3, seed=4, start=60)
    targets["x"] = np.full((3, 416), 0.5, np.float32)

    def propose(i):                                        # rank 0 is the true host; the last target gets no candidate
        return [] if i == 2 else [dict(draw=draws[i], tag="truth"), dict(draw=draws[2], tag="other"),
                                  dict(draw=draws[i], tag="again")]

    def simulate(ds, seeds, n_jobs=1):                     # rank r comes back 0.01 r above the target; one simulation fails
        calls.append((list(ds), list(seeds)))
        return [None if j == 5 else np.full(416, 0.5 + 0.01 * (j % 3), np.float32) for j in range(len(ds))]

    with RunLog(tmp_path / "m.jsonl", dict(k=3), key_field="index", tag="closeness") as log:
        return ec.score(propose, targets, sr.M0Model([np.log(1e-4), 0.0, 0.0]), log, simulate), draws


def test_scoring_simulates_each_candidate_once_in_units_of_the_targets_seed_noise(tmp_path):
    calls = []
    records, draws = _scored(tmp_path, calls)
    assert [r["index"] for r in records] == [60, 61, 62] and records[2]["candidates"] == []
    ratios = [[c["ratio"] for c in r["candidates"]] for r in records[:2]]
    assert ratios[0] == pytest.approx([0.0, 1.0, 4.0], rel=1e-4, abs=1e-9)
    assert ratios[1][:2] == pytest.approx([0.0, 1.0], rel=1e-4, abs=1e-9) and ratios[1][2] is None
    first = records[0]["candidates"][0]
    assert first["family_right"] is True and first["tag"] == "truth" and first["width"] == draws[0].width
    assert "draw" not in first and records[0]["m0"] == pytest.approx(1e-4)
    assert first["host"] == ec.host_record(draws[0]) and records[1]["candidates"][2]["host"] == ec.host_record(draws[1])
    (ds, seeds), = calls                                   # one pool call for the whole batch, one seed per candidate
    assert ds == [draws[0], draws[2], draws[0], draws[1], draws[2], draws[1]]
    assert seeds == ec.candidate_seeds(0, 3) + ec.candidate_seeds(1, 3)


def test_scoring_resumes_from_its_log_without_simulating_again(tmp_path):
    calls = []
    first, _ = _scored(tmp_path, calls)
    again, _ = _scored(tmp_path, calls)
    assert len(calls) == 1 and again == first


def test_summary_groups_by_pattern_and_adds_the_stored_spectrum_block_for_the_lookup(tmp_path):
    records, _ = _scored(tmp_path, [])
    s = ec.summary(records, k=3)
    assert s["n_targets"] == 3 and s["best_within_1"] == pytest.approx(2 / 3) and "stored" not in s
    assert sum(g["n"] for g in s["by_group"].values()) == 3
    assert s["first_right_family"] == pytest.approx(2 / 3)
    for r in records:
        for j, c in enumerate(r["candidates"]):
            c["stored_ratio"] = 0.5 + j
    assert ec.summary(records, k=3)["stored"]["first"]["median"] == pytest.approx(0.5)


def test_cli_scores_the_lookup_and_each_model_and_resumes_without_simulating_again(tmp_path, monkeypatch):
    root, out = tmp_path / "data", tmp_path / "out"
    root.mkdir()
    np.savez(root / "shard_00000.npz", **_arrays(200, seed=5, known=(1, 22, 43, 64))[0])     # 180 train, 10 val, 10 test
    for name, seed in (("a", 0), ("b", 1)):
        (tmp_path / name).mkdir()
        _model(seed).save(tmp_path / name / "model.pt")
    (tmp_path / "m0.json").write_text(json.dumps({"model": {"coef": [-8.0, -0.9, 0.3]}}))
    sims = []

    def simulate(draws, seeds, n_jobs=1):
        sims.append(len(draws))
        return [np.full(416, d.width / 22.0, np.float32) for d in draws]

    monkeypatch.setattr(ec.sr, "simulate_candidates", simulate)
    argv = [str(tmp_path / "a"), str(tmp_path / "b"), "--ensemble", "--data", str(root), "--m0", str(tmp_path / "m0.json"),
            "--n", "4", "--k", "2", "--samples", "300", "--jobs", "1", "--threads", "2", "--out", str(out)]
    ec.main(argv)
    rep = json.loads((out / "closeness.json").read_text())
    assert list(rep["methods"]) == ["truth", "lookup", "lookup-known", "a", "b", "ensemble"]
    known = rep["methods"]["lookup-known"]
    assert known["near_known"]["share_of_hosts"] == 1.0 and known["near_known"]["targets_with_one"] == 1.0
    assert rep["methods"]["truth"]["near_known"]["share_of_hosts"] == 0.0
    assert rep["realism"] == dict(materials=list(rl.NAMED), rel_tol=rl.REL_TOL, abs_tol=rl.ABS_TOL)
    assert "near a known material" in (out / "closeness.txt").read_text()
    assert rep["methods"]["truth"]["candidates_per_target"] == 1.0 and rep["methods"]["truth"]["first_right_family"] == 1.0
    assert rep["methods"]["lookup"]["n_targets"] == 4 and "stored" in rep["methods"]["lookup"]
    assert rep["methods"]["lookup"]["candidates_per_target"] == 2.0
    assert rep["settings"]["k"] == 2 and len(rep["targets"]) == 4 and all(i % 20 == 18 for i in rep["targets"])
    assert set(rep["models"]) == {"a", "b", "ensemble"} and len(rep["models"]["a"]["model_sha256"]) == 64
    text = (out / "closeness.txt").read_text()
    assert text.splitlines()[0].startswith("method") and "ensemble" in text
    done = sum(sims)
    assert done > 0
    ec.main(argv)
    assert sum(sims) == done


def test_cli_can_take_its_targets_from_the_val_rows_near_a_known_material(tmp_path, monkeypatch):
    root, out = tmp_path / "data", tmp_path / "out"
    root.mkdir()
    planted = (1, 22, 43, 64, 18, 38, 58, 78)                       # four train rows and four val rows (index % 20 == 18)
    np.savez(root / "shard_00000.npz", **_arrays(200, seed=5, known=planted)[0])
    (tmp_path / "m0.json").write_text(json.dumps({"model": {"coef": [-8.0, -0.9, 0.3]}}))
    monkeypatch.setattr(ec.sr, "simulate_candidates",
                        lambda draws, seeds, n_jobs=1: [np.full(416, d.width / 22.0, np.float32) for d in draws])
    ec.main(["--data", str(root), "--m0", str(tmp_path / "m0.json"), "--targets", "known", "--n", "3", "--k", "2",
             "--jobs", "1", "--out", str(out)])
    rep = json.loads((out / "closeness.json").read_text())
    assert len(rep["targets"]) == 3 and set(rep["targets"]) <= {18, 38, 58, 78}
    assert rep["settings"]["targets"] == "known" and rep["methods"]["truth"]["near_known"]["share_of_hosts"] == 1.0
    # the default benchmark is unchanged and does not carry the key (its logs stay resumable)
    ec.main(["--data", str(root), "--m0", str(tmp_path / "m0.json"), "--n", "3", "--k", "2", "--jobs", "1",
             "--out", str(tmp_path / "plain")])
    plain = json.loads((tmp_path / "plain" / "closeness.json").read_text())
    assert "targets" not in plain["settings"] and plain["targets"] != rep["targets"]


def test_cli_lookup_can_also_search_the_train_rows_of_extra_data(tmp_path, monkeypatch):
    root, more, out = tmp_path / "data", tmp_path / "more", tmp_path / "out"
    root.mkdir(), more.mkdir()
    base = _arrays(200, seed=5, known=(1, 22, 43, 64))[0]              # lookup-known needs train rows near one
    np.savez(root / "shard_00000.npz", **base)
    start = 3_200_000_000
    extra = _arrays(40, seed=9, start=start)[0]
    val_x = base["x"][base["index"] % 20 == 18]
    extra["x"][:10] = val_x                                          # train rows 0..9 of the extra data: the val spectra
    np.savez(more / "shard_00000.npz", **extra)
    (tmp_path / "m0.json").write_text(json.dumps({"model": {"coef": [-8.0, -0.9, 0.3]}}))
    monkeypatch.setattr(ec.sr, "simulate_candidates",
                        lambda draws, seeds, n_jobs=1: [np.full(416, d.width / 22.0, np.float32) for d in draws])
    ec.main(["--data", str(root), "--m0", str(tmp_path / "m0.json"), "--n", "4", "--k", "2", "--jobs", "1",
             "--lookup-extra", str(more), "--out", str(out)])
    rep = json.loads((out / "closeness.json").read_text())
    assert list(rep["methods"]) == ["truth", "lookup", "lookup-known", "lookup-extra"]
    assert rep["methods"]["lookup-extra"]["stored"]["first"]["median"] == 0.0
    assert rep["methods"]["lookup"]["stored"]["first"]["median"] > 0.0
    assert rep["settings"]["lookup_extra"] == [str(more)]
    rows = [json.loads(line) for line in (out / "lookup-extra.jsonl").read_text().splitlines()]
    assert all(r["record"]["candidates"][0]["source_index"] >= start for r in rows)
    assert all(r["cfg"]["n_train"] == 180 + 36 for r in rows)
