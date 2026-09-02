"""
Paper 2, checklist item 9 (full model): transport the frozen 21-variable MIMIC
model to eICU.

Uses the frozen dictionary v1.0 output for the four organ-support flags and the
A1 preprocessing for the 17 continuous variables. The epsilon floor verified in
step 8 is applied to the categorical emissions; without it a crrt=1 window gets
zero likelihood under three of four states.

Only A1 is run. A2 is not executable on this model -- the preserved state's
gcs_motor emission has SD 0.000119 on the z-scale, so under eICU-derived scaling
its own defining value sits 1,256 SD from the emission mean and the state
vanishes. That is a finding, reported in protocol v1.2 section 10.2, not a
failed analysis.

Four scopes, because the two eligibility rules answer different questions:
  full                 all 8,279 patients
  GCS-eligible         hospitals with >=20 patients and GCS in >=50% of windows
  vent-ascertainable   rule N3, hospitals where the invasive-determining
                       interfaces are wired
  both                 the intersection, the strictest reading

Reminder on naming: the column is `vasopressor` for frozen-feature-order
alignment; the variable is VASOACTIVE SUPPORT and includes dobutamine and
milrinone.
"""
import copy
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "results_full_model"; OUT.mkdir(exist_ok=True)
EPS = 1e-4
RNG = np.random.default_rng(42)

FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
lbl = pd.read_csv(ROOT / "04_outputs/tables/state_label_mapping.csv").set_index("raw_state")["name"].to_dict()
model = torch.load(ROOT / ".model_k4.pt", weights_only=False)
floored = copy.deepcopy(model)
for d in floored.distributions:
    for sub in d.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), EPS, 1 - EPS)
        p = p / p.sum(dim=-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False)
        sub._log_probs = torch.log(p)

# ---- assemble the 21-variable eICU matrix -----------------------------------
cont = pd.read_csv(HERE / "windows/timewindow_level_modeling_eicu_A1.csv")
bina = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv")
w = cont.merge(bina, on=["patientunitstayid", "window_idx"], how="left")
missing = [f for f in FEATS if f not in w.columns]
assert not missing, f"missing features: {missing}"
w = w.sort_values(["patientunitstayid", "window_idx"]).reset_index(drop=True)
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
w = w.merge(coh[["patientunitstayid", "hospitalid", "vent_ascertainable"]], on="patientunitstayid")
elig = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
print(f"eICU matrix: {len(w):,} windows x {len(FEATS)} features, "
      f"{w.patientunitstayid.nunique():,} patients")

lab = [floored.predict(torch.tensor(g[FEATS].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
       for _, g in w.groupby("patientunitstayid", sort=False)]
w["state"] = pd.Series(np.concatenate(lab), index=w.index).map(lbl)
w[["patientunitstayid", "window_idx", "state", "gcs_motor_missing_raw",
   "hospitalid", "vent_ascertainable"]].to_csv(OUT / "eicu_state_assignments_full_A1.csv", index=False)

# ---- MIMIC reference, built identically --------------------------------------
mw = pd.read_csv(ROOT / "04_outputs/tables/timewindow_level_modeling.csv").sort_values(["stay_id", "window_idx"])
mlab = [floored.predict(torch.tensor(g[FEATS].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
        for _, g in mw.groupby("stay_id", sort=False)]
mw["state"] = pd.Series(np.concatenate(mlab), index=mw.index).map(lbl)
MZ = mw.groupby("state")[FEATS].mean()
STATES = list(MZ.index)
mprev = (mw.state.value_counts(normalize=True) * 100).reindex(STATES)
md = mw[["stay_id", "window_idx", "state"]].copy(); md["next"] = md.groupby("stay_id")["state"].shift(-1)
mtm = pd.crosstab(md.state, md["next"], normalize="index").reindex(index=STATES, columns=STATES).fillna(0)
mchange = mw.groupby("stay_id")["state"].nunique().gt(1).mean()

rows, per_state = [], []
SCOPES = [("full", w),
          ("GCS-eligible", w[w.patientunitstayid.isin(elig)]),
          ("vent-ascertainable", w[w.vent_ascertainable == 1]),
          ("both", w[w.patientunitstayid.isin(elig) & (w.vent_ascertainable == 1)])]
for name, sub in SCOPES:
    ez = sub.groupby("state")[FEATS].mean().reindex(STATES)
    corrs = {s: float(np.corrcoef(MZ.loc[s], ez.loc[s])[0, 1]) if ez.loc[s].notna().all() else np.nan
             for s in STATES}
    prev = (sub.state.value_counts(normalize=True) * 100).reindex(STATES).fillna(0)
    d = sub[["patientunitstayid", "window_idx", "state"]].copy()
    d["next"] = d.groupby("patientunitstayid")["state"].shift(-1)
    tm = pd.crosstab(d.state, d["next"], normalize="index").reindex(index=STATES, columns=STATES).fillna(0)
    rank = float(pd.Series(mtm.to_numpy().ravel()).corr(pd.Series(tm.to_numpy().ravel()), method="spearman"))
    selfd = max(abs(tm.loc[s, s] - mtm.loc[s, s]) for s in STATES)
    change = d.groupby("patientunitstayid")["state"].nunique().gt(1).mean()
    null = []
    for _ in range(100):
        pz = sub.assign(state=RNG.permutation(sub.state.values)).groupby("state")[FEATS].mean().reindex(STATES)
        null.append(np.nanmean([np.corrcoef(MZ.loc[s], pz.loc[s])[0, 1] for s in STATES]))
    rows.append({"scope": name, "patients": sub.patientunitstayid.nunique(), "windows": len(sub),
                 "min_profile_corr": round(min(corrs.values()), 3),
                 "null_p95": round(float(np.percentile(null, 95)), 3),
                 "min_prevalence_pct": round(float(prev.min()), 1),
                 "transition_rank_corr": round(rank, 3),
                 "max_self_transition_diff": round(selfd, 3),
                 "pct_changing_state": round(change * 100, 1)})
    for s in STATES:
        per_state.append({"scope": name, "state": s.replace(" analogue", ""),
                          "mimic_prev": round(float(mprev[s]), 1),
                          "eicu_prev": round(float(prev[s]), 1),
                          "profile_corr": round(corrs[s], 3)})
res = pd.DataFrame(rows); ps = pd.DataFrame(per_state)
res.to_csv(OUT / "transport_metrics_full.csv", index=False)
ps.to_csv(OUT / "per_state_full.csv", index=False)
print(f"\nMIMIC reference (frozen 21-variable model): "
      + ", ".join(f"{s.split('-')[0][:12]} {mprev[s]:.1f}%" for s in STATES)
      + f" | changing {mchange*100:.1f}%")
print("\n=== full-model transport metrics ===")
print(res.to_string(index=False))
print("\n=== per state ===")
for sc in ["full", "both"]:
    print(f"\n-- {sc} --")
    print(ps[ps.scope == sc][["state", "mimic_prev", "eicu_prev", "profile_corr"]].to_string(index=False))
print("\n=== prespecified thresholds, protocol section 11.2 ===")
for _, r in res.iterrows():
    checks = [("profile corr >=0.80 all states", r.min_profile_corr >= 0.80),
              ("every state >=5% of windows", r.min_prevalence_pct >= 5.0),
              ("transition rank corr >=0.80", r.transition_rank_corr >= 0.80),
              ("self-transitions within +-0.10", r.max_self_transition_diff <= 0.10),
              ("beats permutation null", r.min_profile_corr > r.null_p95),
              ("floor: no state <0.60", r.min_profile_corr >= 0.60)]
    print(f"  {r.scope:<20} " + "  ".join(f"{'PASS' if ok else 'FAIL'}:{n.split()[0]}" for n, ok in checks))
