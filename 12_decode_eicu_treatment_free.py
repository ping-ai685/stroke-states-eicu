"""
Paper 2, step 9b: transport the frozen treatment-free model to eICU.

This is the first eICU state assignment in the study, so protocol v1.1 section 19
applies from here: nothing below reads an outcome field. Mortality and
support-escalation analyses are step 10 and run only after this is archived.

Decodes both preprocessing conditions required by section 10.2:
  A1  frozen MIMIC medians and z-parameters   (primary)
  A2  eICU-derived medians and z-parameters   (calibration companion)

and reports each on the full cohort and on the GCS-eligible subset, against the
prespecified thresholds in section 11.2. Also runs the amendment-I diagnostic --
prevalence split by observed versus imputed GCS -- and the permutation negative
control, so an observed correlation has a null to beat.

Outputs (results_treatment_free/).
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
FP = HERE / "frozen_params_treatment_free"
OUT = HERE / "results_treatment_free"; OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

model = torch.load(FP / ".model_k4_treatment_free.pt", weights_only=False)
mapping = pd.read_csv(FP / "state_label_mapping.csv").set_index("tf_state")["name"].to_dict()
FEATS = pd.read_csv(FP / "feature_order.csv").feature.tolist()
RAWV = [f[:-2] for f in FEATS]
mimic_prof = pd.read_csv(FP / "mimic_reference_profile.csv", index_col=0)
mimic_tm = pd.read_csv(FP / "mimic_reference_transitions.csv", index_col=0)
elig = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)

# MIMIC reference profile on the z-scale, built the same way eICU's will be
mw = pd.read_csv(ROOT / "04_outputs/tables/timewindow_level_modeling.csv").sort_values(["stay_id", "window_idx"])
mlab = [model.predict(torch.tensor(g[FEATS].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
        for _, g in mw.groupby("stay_id", sort=False)]
mw["state"] = pd.Series(np.concatenate(mlab), index=mw.index).map(mapping)
MIMIC_Z = mw.groupby("state")[FEATS].mean()
mimic_change = mw.groupby("stay_id")["state"].nunique().gt(1).mean()
STATES = list(MIMIC_Z.index)

def dynamics(df, key="patientunitstayid"):
    d = df[[key, "window_idx", "state"]].copy()
    d["next"] = d.groupby(key)["state"].shift(-1)
    tm = pd.crosstab(d.state, d["next"], normalize="index").reindex(index=STATES, columns=STATES).fillna(0)
    changed = d.groupby(key)["state"].nunique().gt(1).mean()
    # episode durations
    d["blk"] = (d.state != d.groupby(key)["state"].shift()).cumsum()
    dur = d.groupby(["blk", "state"]).size().reset_index(name="n").groupby("state")["n"].median()
    return tm, changed, dur

rows, details = [], {}
for cond in ["A1", "A2"]:
    w = pd.read_csv(HERE / f"windows/timewindow_level_modeling_eicu_{cond}.csv") \
          .sort_values(["patientunitstayid", "window_idx"])
    lab = [model.predict(torch.tensor(g[FEATS].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
           for _, g in w.groupby("patientunitstayid", sort=False)]
    w["state"] = pd.Series(np.concatenate(lab), index=w.index).map(mapping)
    w.to_csv(OUT / f"eicu_state_assignments_{cond}.csv", index=False,
             columns=["patientunitstayid", "window_idx", "state", "gcs_motor_missing_raw"])
    details[cond] = w

    for scope, sub in [("full", w), ("eligible", w[w.patientunitstayid.isin(elig)])]:
        ez = sub.groupby("state")[FEATS].mean().reindex(STATES)
        corrs = {s: float(np.corrcoef(MIMIC_Z.loc[s], ez.loc[s])[0, 1])
                 if ez.loc[s].notna().all() else np.nan for s in STATES}
        tm, changed, dur = dynamics(sub)
        prev = (sub.state.value_counts(normalize=True) * 100).reindex(STATES).fillna(0)
        mprev = mimic_prof.window_share_pct.reindex(STATES)
        self_d = {s: float(tm.loc[s, s] - mimic_tm.loc[s, s]) for s in STATES}
        # transition rank correlation over the 16 cells
        a = mimic_tm.reindex(index=STATES, columns=STATES).to_numpy().ravel()
        b = tm.to_numpy().ravel()
        rank = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
        # permutation null for the profile correlation
        null = []
        for _ in range(200):
            perm = sub.copy(); perm["state"] = RNG.permutation(perm.state.values)
            pz = perm.groupby("state")[FEATS].mean().reindex(STATES)
            null.append(np.nanmean([np.corrcoef(MIMIC_Z.loc[s], pz.loc[s])[0, 1] for s in STATES]))
        rows.append({
            "condition": cond, "scope": scope, "patients": sub.patientunitstayid.nunique(),
            "windows": len(sub),
            "min_profile_corr": round(min(corrs.values()), 3),
            "mean_profile_corr": round(float(np.mean(list(corrs.values()))), 3),
            "null_mean_profile_corr": round(float(np.mean(null)), 3),
            "null_p95": round(float(np.percentile(null, 95)), 3),
            "min_state_prevalence_pct": round(float(prev.min()), 1),
            "max_abs_prevalence_diff_pp": round(float((prev - mprev).abs().max()), 1),
            "transition_rank_corr": round(rank, 3),
            "max_abs_self_transition_diff": round(max(abs(v) for v in self_d.values()), 3),
            "pct_patients_changing_state": round(changed * 100, 1),
            "mimic_pct_changing": round(mimic_change * 100, 1),
            **{f"prev_{s.split(' analogue')[0][:18]}": round(float(prev[s]), 1) for s in STATES},
            **{f"corr_{s.split(' analogue')[0][:18]}": round(corrs[s], 3) for s in STATES},
        })

res = pd.DataFrame(rows)
res.to_csv(OUT / "transport_metrics.csv", index=False)
cols = ["condition", "scope", "patients", "windows", "min_profile_corr", "null_p95",
        "min_state_prevalence_pct", "transition_rank_corr", "max_abs_self_transition_diff",
        "pct_patients_changing_state", "mimic_pct_changing"]
print("\n=== transport metrics ===")
print(res[cols].to_string(index=False))

print("\n=== prespecified thresholds, protocol v1.1 section 11.2 ===")
for _, r in res.iterrows():
    checks = [("profile corr >=0.80 all states", r.min_profile_corr >= 0.80),
              ("every state >=5% of windows", r.min_state_prevalence_pct >= 5.0),
              ("transition rank corr >=0.80", r.transition_rank_corr >= 0.80),
              ("self-transitions within +-0.10", r.max_abs_self_transition_diff <= 0.10),
              ("beats permutation null", r.min_profile_corr > r.null_p95),
              ("not-met floor: no state <0.60", r.min_profile_corr >= 0.60)]
    print(f"\n{r.condition} / {r.scope}:")
    for name, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")

# amendment I diagnostic
print("\n=== amendment I: prevalence split by observed vs imputed GCS (A1, full) ===")
w = details["A1"]
d = pd.crosstab(w.state, w.gcs_motor_missing_raw.map({0: "GCS measured", 1: "GCS imputed"}),
                normalize="columns") * 100
d["MIMIC"] = mimic_prof.window_share_pct.reindex(d.index)
d["gap_imputed_minus_measured"] = (d["GCS imputed"] - d["GCS measured"]).round(1)
print(d.round(1).to_string())
d.round(2).to_csv(OUT / "amendment_I_prevalence_split.csv")
print(f"\nWrote {OUT}/")
