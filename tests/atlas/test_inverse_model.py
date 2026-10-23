import numpy as np
import torch

from shazam.inverse.model import InverseModel

torch.set_num_threads(2)


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
