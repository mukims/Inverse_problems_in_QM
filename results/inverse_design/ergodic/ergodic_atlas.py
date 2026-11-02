"""ERGO-1 part A: the conductance correlation length and the ergodic hypothesis, measured on atlas clouds that have a
real ensemble (1,000 disorder seeds per density, same transport code and ~2,000-atom device rule as the parent data).

Per cloud, with the spectra cleaned exactly as the model input (round to 3 decimals, despike, clip: to_input_linear):
* ensemble mean mu(E) and spread sd(E) over the seeds; residual z_j(E) = (T_j(E) - mu(E)) / sd(E) on the channels
  where sd > SD_MIN (gaps and fully ballistic channels carry no fluctuation);
* the paper's C(dE) along energy, as a correlation coefficient pooled over seeds and energies: rho(k), k in channels;
  dE_cor = half width at half maximum (rho = 1/2; below one channel it is extrapolated as exp(-k / k0) and flagged);
  tau = 1 + 2 sum rho(k) up to the first rho <= 0; N_eff = support channels / tau;
* ergodic tests on single samples: the energy average of z_j (should go to 0) and of z_j^2 (should go to 1). Their
  spread over seeds, divided by the spread a stationary Gaussian process with this rho would give, is R_mean and
  R_var: about 1 when one spectrum's energy average behaves like an ensemble average, >> 1 when spectra differ as a
  whole (rare resonances, localisation);
* the part energy averaging cannot remove: `plateau` = mean rho(k) over lags PLATEAU (0.2-0.6 eV), the share of the
  residual variance in the top 1 and top 3 principal components of z over the seeds, and the participation ratio
  (sum lambda)^2 / sum lambda^2 of those eigenvalues (the number of independent modes in one spectrum, the most
  general N_eff);
* what drives the top 3 components: R^2 of a linear fit of their scores on the seed's impurity arrangement (share on
  each sublattice, share within 0.1 of the width from an edge, nearest-neighbour impurity pairs per impurity), from
  the cloud's _config.npz;
* regime: <T> per open pristine channel, var T (diffusive quasi-1D UCF, orthogonal class: 2/15), share of channels
  with <T> < 1;
* the label-free single-spectrum estimators (no ensemble, no pristine) on the first N_CAL seeds, against the truth.

    TRANSMISSION_DATA=... PYTHONPATH=src python results/inverse_design/ergodic/ergodic_atlas.py

Writes atlas_clouds.csv and atlas_calibration.json next to this file.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, minimum_filter1d, uniform_filter1d

import paths
from tbribbon.parent_data import to_input_linear

OUT = Path(__file__).resolve().parent
MATERIALS = ("germanene", "silicene", "hbn", "kagome")     # on the v4-equivalent grid (416 channels of ~0.02 eV)
SD_MIN = 1e-3
K_MAX = 30
N_CAL = 50
WINDOWS = (7, 15, 31)
PLATEAU = slice(10, 31)
EDGE_BAND = 0.1
PAIR_DIST = 1.01                 # bond length 1 in the stored coordinates


def acf_masked(res, mask, k_max=K_MAX):
    """Correlation coefficient of `res` at lags 0..k_max, pooled over rows and over the pairs whose two channels are
    both in `mask` (rows: spectra, columns: energies)."""
    r = np.where(mask, res, 0.0)
    out = np.empty(k_max + 1)
    n = res.shape[1]
    var = (r * r).sum() / max(mask.sum(), 1)
    for k in range(k_max + 1):
        ok = mask[:, k:] & mask[:, :n - k]
        out[k] = (r[:, k:] * r[:, :n - k] * ok).sum() / max(ok.sum(), 1) / var
    return out


def corr_summary(rho):
    """(HWHM in channels, sub-grid flag, tau) of a correlation function rho(0..)."""
    below = np.flatnonzero(rho < 0.5)
    k = int(below[0]) if below.size else len(rho) - 1
    if k == 1:
        hwhm, sub = (np.log(2) / np.log(1 / rho[1]) if 0 < rho[1] < 1 else 0.0), True
    else:
        hwhm, sub = (k - 1) + (rho[k - 1] - 0.5) / (rho[k - 1] - rho[k]), False
    stop = np.flatnonzero(rho[1:] <= 0)
    stop = int(stop[0]) + 1 if stop.size else len(rho)
    tau = 1 + 2 * rho[1:stop].sum()
    return float(hwhm), sub, float(tau)


def label_free(T, window, kind="median"):
    """Single-spectrum correlation function with no ensemble and no pristine spectrum: residual from a running median
    (or mean) of `window` channels, pairs kept only where the whole window lies inside the spectrum's non-zero
    support (so band edges and gaps do not leak into the trend)."""
    support = T > 0
    trend = (median_filter(T, size=(1, window), mode="nearest") if kind == "median"
             else uniform_filter1d(T, window, axis=1, mode="nearest"))
    inside = minimum_filter1d(support.astype(np.uint8), window, axis=1, mode="constant") > 0
    return acf_masked(T - trend, inside)


def gaussian_spread(rho, n):
    """Std over samples of the energy mean of z and of z^2 for a stationary Gaussian process with correlation rho on n
    channels (finite-n sums over lags)."""
    k = np.arange(1, min(len(rho), n))
    w = (n - k) / n
    var_m = (1 + 2 * (w * rho[k]).sum()) / n
    var_v = 2 * (1 + 2 * (w * rho[k] ** 2).sum()) / n
    return np.sqrt(var_m), np.sqrt(var_v)


def arrangement_features(config):
    """Per seed: share of impurities on each sublattice but the last, share within EDGE_BAND of the width from an
    edge, and nearest-neighbour impurity pairs per impurity (columns of a [seeds, features] array)."""
    c = config["coords"].astype(np.float64)
    across = int(np.argmin(np.ptp(c.reshape(-1, 2), axis=0)))      # the short axis of the ribbon
    y = c[..., across]
    u = (y - y.min()) / max(np.ptp(y), 1e-9)
    edge = np.mean(np.minimum(u, 1 - u) < EDGE_BAND, axis=1)
    d = np.sqrt(((c[:, :, None, :] - c[:, None, :, :]) ** 2).sum(-1))
    n = c.shape[1]
    pairs = (np.triu(d < PAIR_DIST, 1)).sum((1, 2)) / n
    sub = config["sublattice"]
    levels = np.unique(sub)
    shares = [np.mean(sub == v, axis=1) for v in levels[:-1]]
    return np.column_stack(shares + [edge, pairs])


def r2(y, F):
    F = np.column_stack([np.ones(len(F)), F])
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    return float(1 - np.var(y - F @ coef) / np.var(y))


def cloud_stats(X, pristine, features=None):
    mu, sd = X.mean(0), X.std(0)
    mask1 = sd > SD_MIN
    Z = np.where(mask1, (X - mu) / np.where(mask1, sd, 1), 0.0)
    mask = np.broadcast_to(mask1, X.shape)
    rho = acf_masked(Z, mask)
    hwhm, sub, tau = corr_summary(rho)
    n = int(mask1.sum())
    m_j = Z[:, mask1].mean(1)
    v_j = (Z[:, mask1] ** 2).mean(1)
    gm, gv = gaussian_spread(rho, n)
    open_ = mask1 & (pristine > 0.5)
    Zs = Z[:, mask1]
    U, S, _ = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    lam = S ** 2
    out = dict(rho=rho, hwhm=hwhm, subgrid=sub, tau=tau, n_support=n, n_eff=n / tau,
               plateau=float(rho[PLATEAU].mean()), pc1_share=float(lam[0] / lam.sum()),
               pc3_share=float(lam[:3].sum() / lam.sum()), participation=float(lam.sum() ** 2 / (lam ** 2).sum()),
               R_mean=float(m_j.std() / gm), R_var=float(v_j.std() / gv),
               g_per_channel=float(np.mean(mu[open_] / pristine[open_])) if open_.any() else np.nan,
               mean_T=float(mu[mask1].mean()), var_T=float((sd[mask1] ** 2).mean()),
               frac_mean_below_1=float(np.mean(mu[mask1] < 1)))
    if features is not None:
        scores = U[:, :3] * S[:3]
        out.update({f"pc{i + 1}_r2_arrangement": r2(scores[:, i], features) for i in range(3)})
        out["mean_r2_arrangement"] = r2(m_j, features)
    return out


def main():
    t0 = time.time()
    root = paths.data_root()
    rows = [r for r in csv.DictReader(open(root / "INDEX.csv")) if r["material"] in MATERIALS and r["n_densities"]]
    table, cal = [], {f"{kind}{w}": [] for kind in ("median", "mean") for w in WINDOWS}
    truth = []
    for r in rows:
        folder = root / r["model_id"]
        E = np.load(folder / "energies_t.npy")
        pristine = np.round(np.load(folder / "pristine.npy").astype(np.float64), 3)
        meta = json.loads((folder / "meta.json").read_text())
        for f in sorted(folder.glob("cloud_d*.npy")):
            if f.name.endswith("_seeds.npy"):
                continue
            X = to_input_linear(np.load(f)).astype(np.float64)
            if len(X) < 200:
                continue
            cfg_path = f.with_name(f.stem + "_config.npz")
            config = dict(np.load(cfg_path)) if cfg_path.exists() else None
            if config is not None and not np.array_equal(config["seeds"], np.load(f.with_name(f.stem + "_seeds.npy"))):
                raise ValueError(f"{cfg_path}: seeds do not match the cloud rows")
            st = cloud_stats(X, pristine, arrangement_features(config) if config is not None else None)
            density = float(f.stem.split("_d")[1])
            table.append(dict(model_id=r["model_id"], material=r["material"], edge=r["edge"], width=int(r["width"]),
                              n_cells=int(r["n_cells"] or 0), density=density, seeds=len(X),
                              n_imp=int(config["n_impurities"]) if config is not None else -1,
                              step_t=float(E[1] - E[0]), **{k: v for k, v in st.items() if k != "rho"},
                              **{f"rho{k}": float(st["rho"][k]) for k in (1, 2, 3, 5, 10)}))
            truth.append(st["rho"][:K_MAX + 1])
            for kind in ("median", "mean"):
                for w in WINDOWS:
                    cal[f"{kind}{w}"].append(label_free(X[:N_CAL], w, kind))
        print(f"{r['model_id']}: {len(table)} clouds ({time.time() - t0:.0f} s)", flush=True)
    with open(OUT / "atlas_clouds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    truth = np.array(truth)
    t_sum = np.array([corr_summary(t) for t in truth])
    report = {}
    for name, est in cal.items():
        est = np.array(est)
        e_sum = np.array([corr_summary(e) for e in est])
        report[name] = dict(
            rho1_bias=float(np.mean(est[:, 1] - truth[:, 1])), rho1_mae=float(np.mean(np.abs(est[:, 1] - truth[:, 1]))),
            rho1_corr=float(np.corrcoef(est[:, 1], truth[:, 1])[0, 1]),
            tau_log_bias=float(np.mean(np.log(e_sum[:, 2] / t_sum[:, 2]))),
            tau_log_corr=float(np.corrcoef(np.log(e_sum[:, 2]), np.log(t_sum[:, 2]))[0, 1]))
    (OUT / "atlas_calibration.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"wrote atlas_clouds.csv ({len(table)} clouds) and atlas_calibration.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
