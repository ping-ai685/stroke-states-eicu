"""
Paper 2, step 8b: the GCS point-mass emission, and what it does to transport.

Found while verifying the epsilon floor (07). The -7.2e6 log-density that turned
up in that script's probe is not caused by the floor -- it is a property of the
published model, and it matters more for eICU than the CRRT zero does.

In the "neurologically preserved-low support" state, the gcs_motor emission has
a fitted SD of 0.000203 on the z-scale. That is not a Normal, it is a point mass
at motor = 6. The state holds 63.3% of MIMIC windows and 100.0% of them have
motor = 6: the emission is a hard membership gate.

The transport hazard is the interaction with imputation. MIMIC's train median
for gcs_motor is 6, so a window with unmeasured GCS is imputed to exactly the
point mass -- it lands 0.00 SD from the emission mean. In MIMIC that affects
4.9% of the state's windows. In eICU, GCS is missing in 46.5% of windows
(18.8% in the eligible subset), and every one of those will satisfy the gate.

This does not force assignment -- 21 other features and the transition structure
still apply, and 21.1% of MIMIC's motor=6 windows are assigned elsewhere -- but
it removes the single strongest exclusion criterion for the largest state, in
proportion to how little GCS a hospital charts. That confounds Aim 3 directly:
a hospital that charts less GCS will appear to have more preserved-state
patients.

The underlying cause is modelling an ordinal 1-6 subscore with a Normal
emission, which feature_spec.py already flags as future work. The published
model cannot be changed here; the response is measurement, not repair.

Outputs (audit/): gcs_emission_diagnostic.csv
"""
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401
HERE = Path(__file__).parent
OUT = HERE / "audit"

model = torch.load(ROOT / ".model_k4.pt", weights_only=False)
F = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
CONT = [f for f in F if f.endswith("_z")]
lbl = pd.read_csv(ROOT / "04_outputs/tables/state_label_mapping.csv").set_index("raw_state")
imp = pd.read_csv(HERE / "frozen_params/imputation_medians_train.csv").set_index("variable")
zp = pd.read_csv(HERE / "frozen_params/zscore_params_train.csv", index_col=0)

# 1. how tight is every continuous emission?
rows = []
for k, d in enumerate(model.distributions):
    for name, sub in zip(CONT, d.distributions[:len(CONT)]):
        rows.append({"state": lbl.loc[k, "name"], "feature": name,
                     "mean_z": float(sub.means.detach().ravel()[0]),
                     "sd_z": float(torch.sqrt(sub.covs.detach().ravel())[0])})
sd = pd.DataFrame(rows)
print("Tightest continuous emissions (sd on the z-scale; 1.0 = one MIMIC training SD):")
print(sd.nsmallest(5, "sd_z").to_string(index=False))
print(f"\nemissions with sd < 0.01: {(sd.sd_z < 0.01).sum()} of {len(sd)}")

# 2. does the imputed value sit on the point mass?
med = imp.loc["gcs_motor", "train_median_post_ffill"]
zmed = (med - zp.loc["gcs_motor", "mean"]) / zp.loc["gcs_motor", "std"]
pt = sd[(sd.state == "neurologically preserved-low support") & (sd.feature == "gcs_motor_z")].iloc[0]
print(f"\nMIMIC train median gcs_motor = {med:.0f} -> z = {zmed:.6f}")
print(f"preserved-state emission      mean z = {pt.mean_z:.6f}, sd = {pt.sd_z:.6f}")
print(f"imputed value sits {abs(zmed - pt.mean_z) / pt.sd_z:.2f} SD from the emission mean")
for mv in (5.0, 4.0):
    zz = (mv - zp.loc["gcs_motor", "mean"]) / zp.loc["gcs_motor", "std"]
    print(f"  log-density for gcs_motor={mv:.0f} under that state: {-0.5*((zz-pt.mean_z)/pt.sd_z)**2:,.0f}")

# 3. decode MIMIC and confirm the gate empirically
w = pd.read_csv(ROOT / "04_outputs/tables/timewindow_level_modeling.csv").sort_values(["stay_id", "window_idx"])
lab = [model.predict(torch.tensor(g[F].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
       for _, g in w.groupby("stay_id", sort=False)]
w["state"] = pd.Series(np.concatenate(lab), index=w.index).map(lbl.name)
p = w[w.state == "neurologically preserved-low support"]
share6 = (p.gcs_motor == 6).mean() * 100
elsewhere6 = (w[w.state != "neurologically preserved-low support"].gcs_motor == 6).mean() * 100
print(f"\nPreserved state: {len(p):,} windows, {share6:.1f}% have gcs_motor = 6 "
      f"(necessary), while {elsewhere6:.1f}% of non-preserved windows also have 6 (not sufficient)")
print(f"Of the preserved state's windows, {p.gcs_motor_missing_raw.mean()*100:.1f}% had gcs_motor imputed")

pd.DataFrame([
    {"finding": "preserved-state gcs_motor emission SD (z)", "value": round(pt.sd_z, 6)},
    {"finding": "imputed value distance from emission mean (SD)", "value": round(abs(zmed-pt.mean_z)/pt.sd_z, 3)},
    {"finding": "preserved-state windows with gcs_motor=6 (%)", "value": round(share6, 1)},
    {"finding": "non-preserved windows with gcs_motor=6 (%)", "value": round(elsewhere6, 1)},
    {"finding": "preserved-state windows with imputed gcs_motor, MIMIC (%)",
     "value": round(p.gcs_motor_missing_raw.mean()*100, 1)},
    {"finding": "eICU windows with no measured GCS, full cohort (%)", "value": 46.5},
    {"finding": "eICU windows with no measured GCS, eligible subset (%)", "value": 18.8},
]).to_csv(OUT / "gcs_emission_diagnostic.csv", index=False)
print(f"\nWrote {OUT}/gcs_emission_diagnostic.csv")
