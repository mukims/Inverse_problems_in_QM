import numpy as np
import pytest
import torch

from shazam.inverse.model import InverseModel


@pytest.fixture(autouse=True)
def two_threads():
    """Small models: two threads are enough; the previous setting is restored for the rest of the session."""
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def _problem():
    n_states, P = 3, torch.randn(3 + 14, 416)                     # spectrum = fixed projection of (state one-hot, z) + noise
    masks = torch.tensor([[1] * 4 + [0] * 10, [1] * 8 + [0] * 6, [1] * 14], dtype=torch.bool)

    def batch(B):
        s = torch.randint(0, n_states, (B,)); z = torch.randn(B, 14) * masks[s]
        x = torch.cat([torch.nn.functional.one_hot(s, 3).float(), z], 1) @ P / 10 + 0.01 * torch.randn(B, 416)
        return x, s, z, masks[s]
    return n_states, masks, batch


def test_learns_a_synthetic_inverse_problem():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    n_states, masks, batch = _problem()
    m = InverseModel(n_states); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(600):
        loss, _ = m.loss(*batch(256)); opt.zero_grad(); loss.backward(); opt.step()
    x, s, z, mk = batch(1)
    states, zs = m.sample(x, 500)
    assert (states == s[0]).float().mean() > 0.9
    sel = states == s[0]
    assert torch.all(zs[sel][:, ~mk[0]] == 0) and (zs[sel][:, mk[0]].mean(0) - z[0, mk[0]]).abs().max() < 0.3


def test_inactive_slots_are_exactly_zero_for_every_sample_and_state_can_be_fixed():
    torch.manual_seed(1)
    n_states, masks, batch = _problem()
    m = InverseModel(n_states); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(5):
        loss, parts = m.loss(*batch(32)); opt.zero_grad(); loss.backward(); opt.step()
    assert set(parts) >= {"ce", "flow"}
    x, *_ = batch(1)
    for k in range(n_states):
        states, zs = m.sample(x, 20, steps=4, state=k)
        assert torch.all(states == k) and zs.shape == (20, 14) and not zs.requires_grad
        assert torch.all(zs[:, ~masks[k]] == 0)
    states, zs = m.sample(x, 200, steps=4)                         # mixed states: each row obeys its own mask
    assert torch.all(zs[~masks[states]] == 0)


def test_save_load_round_trip(tmp_path):
    torch.manual_seed(2)
    n_states, masks, batch = _problem()
    m = InverseModel(n_states, state_masks=masks)
    loss, _ = m.loss(*batch(16)); loss.backward()
    m.save(tmp_path / "m.pt")
    m2 = InverseModel.load(tmp_path / "m.pt")
    assert not m2.training
    x = batch(1)[0]
    m.eval()
    torch.manual_seed(5); a = m.sample(x, 10, steps=4)
    torch.manual_seed(5); b = m2.sample(x, 10, steps=4)
    assert torch.equal(a[0], b[0]) and torch.allclose(a[1], b[1])
    # a model that learned its masks from training batches keeps them too
    m3 = InverseModel(n_states); m3.loss(*batch(64)); m3.save(tmp_path / "m3.pt")
    m4 = InverseModel.load(tmp_path / "m3.pt")
    assert torch.equal(m3.state_masks, m4.state_masks) and torch.equal(m3.mask_known, m4.mask_known)


def test_group_norm_can_be_dropped_and_the_choice_survives_save_load(tmp_path):
    n_states, masks, batch = _problem()
    m = InverseModel(n_states, state_masks=masks, group_norm=False)
    assert not any(isinstance(l, torch.nn.GroupNorm) for l in m.encoder)
    assert m.encode(batch(4)[0]).shape == (4, 128)
    m.save(tmp_path / "m.pt")
    assert not InverseModel.load(tmp_path / "m.pt").cfg["group_norm"]
    # a checkpoint written before the option existed (no "group_norm" in its cfg) loads with GroupNorm, as it was trained
    old = InverseModel(n_states, state_masks=masks)
    ck = {"cfg": {k: v for k, v in old.cfg.items() if k != "group_norm"}, "state_dict": old.state_dict()}
    torch.save(ck, tmp_path / "old.pt")
    assert sum(isinstance(l, torch.nn.GroupNorm) for l in InverseModel.load(tmp_path / "old.pt").encoder) == 3


def test_fixed_mask_loss_rejects_a_mismatching_mask_and_sample_checks_the_input_shape():
    torch.manual_seed(3)
    n_states, masks, batch = _problem()
    m = InverseModel(n_states, state_masks=masks)
    x, s, z, mk = batch(8)
    m.loss(x, s, z, mk)                                            # the matching mask is accepted
    with pytest.raises(ValueError, match="state_masks"):
        m.loss(x, s, z, ~mk)
    x1 = batch(1)[0]
    with pytest.raises(ValueError, match=r"\(1, 416\)"):
        m.sample(x1[0], 3)                                         # 1-D
    with pytest.raises(ValueError, match=r"\(1, 416\)"):
        m.sample(batch(2)[0], 3)                                   # two spectra
    with pytest.raises(ValueError, match=r"\(1, 416\)"):
        m.sample(x1[:, :400], 3)
    torch.manual_seed(5); a = m.sample(x1, 5, steps=4)
    torch.manual_seed(5); b = m.sample(x1.numpy().astype(np.float64), 5, steps=4)    # numpy float64 is cast to float32
    assert torch.equal(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_the_input_transform_is_recorded_and_old_checkpoints_read_as_log(tmp_path):
    n_states, masks, _ = _problem()
    m = InverseModel(n_states, state_masks=masks, input="linear")
    m.save(tmp_path / "m.pt")
    assert InverseModel.load(tmp_path / "m.pt").cfg["input"] == "linear"
    old = InverseModel(n_states, state_masks=masks)
    ck = {"cfg": {k: v for k, v in old.cfg.items() if k != "input"}, "state_dict": old.state_dict()}
    torch.save(ck, tmp_path / "old.pt")
    assert InverseModel.load(tmp_path / "old.pt").cfg["input"] == "log"


def test_extra_conv_blocks_add_stride_one_depth_and_old_checkpoints_have_none(tmp_path):
    n_states, masks, batch = _problem()
    base, deep = InverseModel(n_states, state_masks=masks), InverseModel(n_states, state_masks=masks, extra_conv=1)
    convs = [l for l in deep.encoder if isinstance(l, torch.nn.Conv1d)]
    assert len(convs) == 4 and convs[-1].stride == (1,) and convs[-1].in_channels == convs[-1].out_channels == 128
    assert deep.encode(batch(3)[0]).shape == (3, 128)
    n = lambda m: sum(p.numel() for p in m.encoder.parameters())
    assert n(deep) - n(base) == 128 * 128 * 3 + 128 + 2 * 128               # one conv plus its GroupNorm
    deep.save(tmp_path / "d.pt")
    assert InverseModel.load(tmp_path / "d.pt").cfg["extra_conv"] == 1
    ck = {"cfg": {k: v for k, v in base.cfg.items() if k != "extra_conv"}, "state_dict": base.state_dict()}
    torch.save(ck, tmp_path / "old.pt")
    assert InverseModel.load(tmp_path / "old.pt").cfg["extra_conv"] == 0


def test_mlp_encoder_replaces_the_convolutions_and_keeps_the_heads(tmp_path):
    n_states, masks, batch = _problem()
    m = InverseModel(n_states, state_masks=masks, encoder="mlp")
    assert not any(isinstance(l, torch.nn.Conv1d) for l in m.encoder)
    assert sum(isinstance(l, torch.nn.LayerNorm) for l in m.encoder) == 2
    assert m.encode(batch(5)[0]).shape == (5, 128)
    n = sum(p.numel() for p in m.encoder.parameters())
    assert n == (416 * 512 + 512) + 2 * 512 + (512 * 256 + 256) + 2 * 256 + (256 * 128 + 128)
    cnn = InverseModel(n_states, state_masks=masks)
    assert [p.shape for p in m.discrete.parameters()] == [p.shape for p in cnn.discrete.parameters()]
    loss, _ = m.loss(*batch(16)); loss.backward()                         # trains end to end
    m.save(tmp_path / "m.pt")
    assert InverseModel.load(tmp_path / "m.pt").cfg["encoder"] == "mlp"
    ck = {"cfg": {k: v for k, v in cnn.cfg.items() if k != "encoder"}, "state_dict": cnn.state_dict()}
    torch.save(ck, tmp_path / "old.pt")
    assert InverseModel.load(tmp_path / "old.pt").cfg["encoder"] == "cnn"
    with pytest.raises(ValueError):
        InverseModel(n_states, encoder="mlp", extra_conv=1)
    with pytest.raises(ValueError):
        InverseModel(n_states, encoder="transformer")


def test_mlp_hidden_and_dropout_are_built_and_recorded(tmp_path):
    n_states, masks, batch = _problem()
    m = InverseModel(n_states, state_masks=masks, encoder="mlp", mlp_hidden=(64, 32, 16), dropout=0.1)
    assert sum(isinstance(l, torch.nn.LayerNorm) for l in m.encoder) == 3
    assert sum(isinstance(l, torch.nn.Dropout) for l in m.encoder) == 3
    m.save(tmp_path / "m.pt")
    m2 = InverseModel.load(tmp_path / "m.pt")
    assert m2.cfg["mlp_hidden"] == [64, 32, 16] and m2.cfg["dropout"] == 0.1
    x = batch(4)[0]
    with torch.no_grad():
        assert torch.equal(m2.encode(x), m2.encode(x))                  # load() returns eval mode: dropout is off


def test_cnn_with_bandtop_input_convolves_the_spectrum_and_adds_the_band_top_separately(tmp_path):
    n_states, masks, batch = _problem()
    m = InverseModel(n_states, channels=417, state_masks=masks, input="bandtop", extra_conv=1)
    x = torch.cat([batch(4)[0], torch.rand(4, 1)], 1)                       # 416 spectrum values + the band top
    h = m.encode(x)
    assert h.shape == (4, 128)
    x2 = x.clone(); x2[:, -1] += 0.3                                         # only the band top changes
    with torch.no_grad():
        d = m.encode(x2) - m.encode(x)
        assert torch.allclose(d, m.band_top(x2[:, -1:]) - m.band_top(x[:, -1:]), atol=1e-6)
    m.save(tmp_path / "m.pt")
    assert torch.equal(InverseModel.load(tmp_path / "m.pt").encode(x), m.eval().encode(x))
    mlp = InverseModel(n_states, channels=417, state_masks=masks, input="bandtop", encoder="mlp")
    assert not hasattr(mlp, "band_top") and mlp.encode(x).shape == (4, 128)  # the MLP reads all 417 values directly


def test_spectral_pool_keeps_the_mean_and_low_frequencies():
    from shazam.inverse.model import SpectralPool1d
    pool = SpectralPool1d(13)
    const = torch.full((2, 3, 52), 2.5)
    assert torch.allclose(pool(const), torch.full((2, 3, 13), 2.5), atol=1e-6)
    t = torch.arange(52) / 52
    wave = torch.cos(2 * torch.pi * 2 * t).expand(1, 1, 52)                       # 2 cycles: well below the cut-off
    expect = torch.cos(2 * torch.pi * 2 * torch.arange(13) / 13)
    assert torch.allclose(pool(wave)[0, 0], expect, atol=1e-5)
    fast = torch.cos(2 * torch.pi * 20 * t).expand(1, 1, 52)                      # 20 cycles: removed
    assert pool(fast).abs().max() < 1e-5


@pytest.mark.parametrize("pool, n_linear_in", [("avg13", 128 * 13), ("gap", 128), ("spectral", 128 * 13)])
def test_pool_options_build_and_are_recorded(pool, n_linear_in, tmp_path):
    n_states, masks, batch = _problem()
    m = InverseModel(n_states, channels=417, state_masks=masks, input="bandtop", extra_conv=1, pool=pool)
    assert m.encoder[-1].in_features == n_linear_in
    x = torch.cat([batch(3)[0], torch.rand(3, 1)], 1)
    assert m.encode(x).shape == (3, 128)
    m.save(tmp_path / "m.pt")
    assert InverseModel.load(tmp_path / "m.pt").cfg["pool"] == pool
    old = InverseModel(n_states, state_masks=masks)
    ck = {"cfg": {k: v for k, v in old.cfg.items() if k != "pool"}, "state_dict": old.state_dict()}
    torch.save(ck, tmp_path / "old.pt")
    assert InverseModel.load(tmp_path / "old.pt").cfg["pool"] == "avg13"


def test_ensemble_model_combines_seeds_and_masks_dropped_patterns(tmp_path):
    from shazam.inverse.encoding import STATES
    from shazam.inverse.model import EnsembleInverseModel
    from tbribbon.parent import TYPES

    n_states, masks, batch = _problem()
    m1 = InverseModel(n_states, state_masks=masks)
    m2 = InverseModel(n_states, state_masks=masks)
    m1.save(tmp_path / "m1.pt")
    m2.save(tmp_path / "m2.pt")

    ens = EnsembleInverseModel.load([tmp_path / "m1.pt", tmp_path / "m2.pt"])
    assert ens.cfg["ensemble"] == 2
    x = batch(1)[0]
    states, z = ens.sample(x, 50, steps=4)
    assert states.shape == (50,) and z.shape == (50, 14)
    assert torch.all(z[~masks[states]] == 0)

    # Test drop_patterns with full 156-state model
    n_full = len(STATES)
    full_masks = torch.ones(n_full, 14, dtype=torch.bool)
    f1 = InverseModel(n_full, state_masks=full_masks)
    f2 = InverseModel(n_full, state_masks=full_masks)
    f1.save(tmp_path / "f1.pt")
    f2.save(tmp_path / "f2.pt")

    ens_chk = EnsembleInverseModel.load([tmp_path / "f1.pt", tmp_path / "f2.pt"], drop_patterns=("CHK",))
    chk_states = set(k for k, s in enumerate(STATES) if "".join(t for t, on in zip(TYPES, s.pattern) if on) == "CHK")
    assert len(chk_states) > 0

    x_full = torch.randn(1, 416)
    sampled_states, _ = ens_chk.sample(x_full, 200, steps=2)
    assert not any(int(s) in chk_states for s in sampled_states)

