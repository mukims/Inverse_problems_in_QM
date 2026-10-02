"""Shared energy axis: place a stored spectrum (energies in units of its material's hopping t) on an InputSpec's axis."""
import numpy as np


def axis_scale(spec, model):
    """Spec units per stored unit: the model's hopping t_ev on an eV axis (v3), 1 on a units-of-t axis (v1, v2)."""
    if spec.unit == "eV":
        if not model.t_ev or model.t_ev <= 0:
            raise ValueError(f"{model.model_id} has no physical hopping t_ev ({model.t_ev}); it cannot go on an eV axis")
        return float(model.t_ev)
    return 1.0


def on_axis(spec, model, e_t, band_top_t=None):
    s = axis_scale(spec, model)
    return np.asarray(e_t, dtype=float) * s, (None if band_top_t is None else float(band_top_t) * s)


def generation_grid_t(spec, model):
    """The spec's channel energies in the model's own units of t, so newly generated spectra land exactly on the axis."""
    return spec.energies_t() / axis_scale(spec, model)
