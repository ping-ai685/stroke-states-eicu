"""
Paper 2, step 7b: apply the FROZEN MIMIC preprocessing to the eICU window table.

Protocol v1.1 section 9. Nothing here is estimated from eICU in the A1 path:
forward-fill limits, residual imputation medians and z-score parameters all come
from frozen_params/, which 00_freeze_mimic_parameters.py wrote and which are
never recomputed.

Two outputs, because section 10.2 requires both:
  A1  frozen MIMIC medians and MIMIC z-parameters   (primary)
  A2  frozen structure, eICU-derived medians and z-parameters (companion, to
      separate a case-mix difference from a measurement-calibration shift)

`{var}_missing_raw` is written for every variable, as the discovery pipeline
does, because amendment I makes the observed-versus-imputed split a required
diagnostic: a window whose GCS was imputed lands exactly on the preserved
state's point-mass emission, so state prevalence has to be reportable
separately for the two groups.

Outputs (windows/):
  timewindow_level_modeling_eicu_A1.csv
  timewindow_level_modeling_eicu_A2.csv
  preprocessing_qa.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "windows"
FP = HERE / "frozen_params"

VITAL_TIER = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c",
              "gcs_eye", "gcs_motor", "urine_output_ml"]
LAB_TIER = ["wbc", "hemoglobin", "platelet", "creatinine", "bun", "sodium", "potassium", "glucose"]
ALL = VITAL_TIER + LAB_TIER                     # 17 continuous; gcs_verbal is excluded by design

imp = pd.read_csv(FP / "imputation_medians_train.csv").set_index("variable")
zp = pd.read_csv(FP / "zscore_params_train.csv", index_col=0)
ff = pd.read_csv(FP / "ffill_limits.csv").set_index("variable").ffill_limit_windows.to_dict()

raw = pd.read_csv(OUT / "timewindow_level_raw_eicu.csv").sort_values(
    ["patientunitstayid", "window_idx"]).reset_index(drop=True)
print(f"raw eICU window table: {len(raw):,} windows, {raw.patientunitstayid.nunique():,} patients")

for v in ALL:
    raw[f"{v}_missing_raw"] = raw[v].isna().astype(int)

def build(label, medians, means, sds):
    d = raw.copy()
    for v in ALL:
        d[v] = d.groupby("patientunitstayid")[v].ffill(limit=ff[v])
        d[f"{v}_ffilled"] = (d[f"{v}_missing_raw"] & d[v].notna()).astype(int)
        d[v] = d[v].fillna(medians[v])
        d[f"{v}_z"] = (d[v] - means[v]) / sds[v]
    d["gcs_em"] = d.gcs_eye + d.gcs_motor
    d.to_csv(OUT / f"timewindow_level_modeling_eicu_{label}.csv", index=False)
    return d

# ---- A1: everything frozen from MIMIC --------------------------------------
a1 = build("A1",
           {v: imp.loc[v, "train_median_post_ffill"] for v in ALL},
           {v: zp.loc[v, "mean"] for v in ALL},
           {v: zp.loc[v, "std"] for v in ALL})

# ---- A2: eICU-derived medians and scaling ----------------------------------
e_med, e_mean, e_sd = {}, {}, {}
for v in ALL:
    filled = raw.groupby("patientunitstayid")[v].ffill(limit=ff[v])
    e_med[v] = float(filled.median())
    s = filled.fillna(e_med[v])
    e_mean[v], e_sd[v] = float(s.mean()), float(s.std())
a2 = build("A2", e_med, e_mean, e_sd)

# ---- QA --------------------------------------------------------------------
rows = []
for v in ALL:
    rows.append({
        "variable": v,
        "pct_measured": round((1 - raw[f"{v}_missing_raw"].mean()) * 100, 1),
        "pct_after_ffill": round(a1[f"{v}_ffilled"].mean() * 100 + (1 - raw[f"{v}_missing_raw"].mean()) * 100, 1),
        "pct_median_imputed": round((raw[f"{v}_missing_raw"].mean() - a1[f"{v}_ffilled"].mean()) * 100, 1),
        "mimic_median": round(float(imp.loc[v, "train_median_post_ffill"]), 2),
        "eicu_median": round(e_med[v], 2),
        "mimic_mean": round(float(zp.loc[v, "mean"]), 2),
        "eicu_mean": round(e_mean[v], 2),
        "mimic_sd": round(float(zp.loc[v, "std"]), 2),
        "eicu_sd": round(e_sd[v], 2),
    })
qa = pd.DataFrame(rows)
qa["mean_shift_in_mimic_sd"] = ((qa.eicu_mean - qa.mimic_mean) / qa.mimic_sd).round(3)
qa.to_csv(OUT / "preprocessing_qa.csv", index=False)
print("\n" + qa.to_string(index=False))
print(f"\nLargest calibration shifts (eICU mean minus MIMIC mean, in MIMIC SD units):")
print(qa.reindex(qa.mean_shift_in_mimic_sd.abs().sort_values(ascending=False).index)
        [["variable", "mean_shift_in_mimic_sd"]].head(6).to_string(index=False))
print(f"\nA1 z-columns written: {sum(1 for c in a1.columns if c.endswith('_z'))}")
