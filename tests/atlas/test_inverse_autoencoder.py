import pytest
import torch

from shazam.inverse.autoencoder import SpectrumAutoencoder
from shazam.inverse.model import InverseModel


@pytest.fixture(autouse=True)
def two_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def _batch(B, P):
    """Band-top-shaped input (416 values + a band top in (0, 1]) and a target that is a fixed function of it."""
    z = torch.randn(B, 6)
    x = torch.cat([torch.relu(z @ P[0]), torch.rand(B, 1) * 0.9 + 0.1], 1)
    return x, torch.relu(z @ P[1])


@pytest.mark.parametrize("kind", ["cnn", "mlp"])
def test_autoencoder_encodes_like_the_inverse_model_and_copies_its_encoder(kind, tmp_path):
    torch.manual_seed(0)
    P = (torch.randn(6, 416), torch.randn(6, 416))
    ae = SpectrumAutoencoder(encoder=kind, extra_conv=1 if kind == "cnn" else 0, mlp_hidden=(64, 32), dropout=0.0)
    x, y = _batch(8, P)
    assert ae(x).shape == (8, 416) and ae.encode(x).shape == (8, 128)
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    first = ae.loss(*_batch(256, P)).item()
    for _ in range(300):
        loss = ae.loss(*_batch(256, P)); opt.zero_grad(); loss.backward(); opt.step()
    assert ae.loss(*_batch(256, P)).item() < 0.5 * first
    model = InverseModel(3, **ae.inverse_kwargs())
    ae.copy_encoder_into(model)
    ae.eval(); model.eval()
    with torch.no_grad():
        assert torch.allclose(ae.encode(x), model.encode(x), atol=1e-6)
    ae.save(tmp_path / "ae.pt")
    ae2 = SpectrumAutoencoder.load(tmp_path / "ae.pt")
    with torch.no_grad():
        assert torch.equal(ae2(x), ae(x))
    wrong = dict(ae.inverse_kwargs(), mlp_hidden=(16,)) if kind == "mlp" else dict(ae.inverse_kwargs(), extra_conv=0)
    with pytest.raises(ValueError, match="mlp_hidden" if kind == "mlp" else "extra_conv"):
        ae.copy_encoder_into(InverseModel(3, **wrong))


def test_cnn_encoder_copies_into_a_model_with_default_mlp_settings():
    ae = SpectrumAutoencoder(encoder="cnn", extra_conv=1, mlp_hidden=(1024, 512), dropout=0.0)
    model = InverseModel(3, channels=417, input="bandtop", extra_conv=1)          # mlp_hidden default (512, 256)
    ae.copy_encoder_into(model)                                                     # mlp settings do not apply
    with pytest.raises(ValueError, match="input"):
        ae.copy_encoder_into(InverseModel(3, channels=417, input="linear", extra_conv=1))
