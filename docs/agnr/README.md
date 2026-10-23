# docs/ — Long-form explanations

Prose write-ups of how the models work and why they are built that way. Read these
before diving into the code — they carry the reasoning that docstrings do not.

| Document | Covers | Pairs with |
|---|---|---|
| `walkthrough.md` | **Start here.** End-to-end tour of the inverse-scattering problem and how the pieces fit together. | the whole tree |
| `inverse_model_explanation.md` | The 10×10 distance-matrix formulation, the ResNet encoder/decoder, and the loss design | [`../defect_reconstruction/inverse_model.py`](../defect_reconstruction/inverse_model.py) |
| `patched_transformer_explanation.md` | Why patching a 1D spectrum and using global self-attention beats a local-receptive-field CNN for correlating distant spectral features | [`../defect_reconstruction/patched_transformer_model.py`](../defect_reconstruction/patched_transformer_model.py), [`../concentration/patched_transformer_v2.py`](../concentration/patched_transformer_v2.py) |

## Related documents elsewhere

| Document | Location | Covers |
|---|---|---|
| `RL_SCHEME.md` | [`../agent/`](../agent/) | MDP formulation of the diagnosis loop (kept with the agent code it specifies) |
| `LOGBOOK.md` | repository root | Build registry, benchmark metrics, bug history and root causes |
| `README.md` | repository root | Project overview, datasets, published benchmark tables |
| `README.md` | [`../`](../) | Code map for this tree — which folder does what |

## A caution on staleness

These documents describe architectures at the time of writing and some have drifted from
the code. In particular, the transformer write-ups describe earlier configurations
(different embedding dims, depth, and patch counts) than
[`../multi_width/mw_transformer.py`](../multi_width/mw_transformer.py) uses today, and
they predate the positional-embedding and warmup changes documented in
[`../multi_width/README.md`](../multi_width/README.md).

**Treat the code as authoritative and these as background.**
