"""Does the upper half of the spectrum (1.5-3 eV) add information about concentration?
Same XGBoost regressor as mw_xgboost.py (500 trees, depth 8, lr 0.04), same seed split as BUILD-09,
only the input energy window changes. Width is given as a feature (the width classifier is ~100% accurate)."""
import sys, time, numpy as np, xgboost as xgb
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import mw_common as mc
D = mc.DEFAULT_DATA_DIR if hasattr(mc, "DEFAULT_DATA_DIR") else "/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data"
tr, _, te, _, _ = mc.load_data(D, samples_per_conc=3000, spectrum_len=300, split="seed")
(Xtr, wtr, ctr, _), (Xte, wte, cte, _) = tr, te
print(f"train {len(Xtr):,} | test {len(Xte):,} (held-out seeds, same test set as BUILD-09)", flush=True)
windows = {"0-1.5 eV (150 ch, current)": slice(0, 150), "1.5-3 eV (150 ch, upper only)": slice(150, 300), "0-3 eV (300 ch, full)": slice(0, 300)}
for name, sl in windows.items():
    t0 = time.time()
    reg = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=6)
    reg.fit(np.hstack([Xtr[:, sl], wtr[:, None]]), ctr)
    p = reg.predict(np.hstack([Xte[:, sl], wte[:, None]])); e = np.abs(p - cte)
    bands = " ".join(f"{lo}-{lo+18}:{e[(cte>=lo)&(cte<lo+20)].mean():.2f}" for lo in (2, 22, 42, 62, 82))
    print(f"{name:<32s} MAE {e.mean():.3f} | 7-AGNR {e[wte==0].mean():.3f} | 9-AGNR {e[wte==1].mean():.3f} | RMSE {np.sqrt(np.mean((p-cte)**2)):.3f} | {time.time()-t0:.0f}s", flush=True)
    print(f"{'':<32s} by c band: {bands}", flush=True)
