"""
Paper 2, step 0: persist every MIMIC-IV parameter that the eICU frozen-transport
analysis (Validation A) has to reuse unchanged.

Why this script has to exist at all: 03_impute_missing.py computes the train-set
imputation medians INLINE and never writes them out, so the Paper 1 pipeline
cannot currently be replayed on a new dataset without re-reading MIMIC. A frozen
external validation is only frozen if the parameters are on disk, dated, and
never recomputed. This script reproduces the Paper 1 imputation step exactly --
same variable tiers, same forward-fill limits, same sequential order (ffill for
one variable, then that variable's train median) -- and saves the result.

Reads only Paper 1 outputs; writes only into 07_paper2_eicu/frozen_params/.
Nothing under 03_code or 04_outputs is touched.

Outputs (frozen_params/):
  imputation_medians_train.csv   train-set median per continuous variable,
                                 computed post-forward-fill (Paper 1 ordering)
  ffill_limits.csv               forward-fill window limit per variable
  zscore_params_train.csv        copy of 04_outputs standardization_params.csv
  emission_binary_frozen.csv     K=4 Bernoulli probabilities for the 4 treatment
                                 flags, with the epsilon floor applied
  transition_frozen.csv          K=4 HMM transition matrix (model parameters)
  start_frozen.csv               K=4 HMM initial-state probabilities
  feature_order.csv              the exact 21-column feature order the model
                                 expects; eICU matrices must match it positionally
"""
import numpy as np
import pandas as pd
import sys
import torch

ROOT = "/Users/pinglei/Desktop/卒中流程预测MIMIC IV data"
sys.path.insert(0, f"{ROOT}/03_code")
import feature_spec

OUT = f"{ROOT}/07_paper2_eicu/frozen_params"
T1 = f"{ROOT}/04_outputs/tables"

# Zero-probability guard. The K=4 model has crrt P(1)=0.000 exactly in three of
# the four states (only the renal-dysfunction state ever sees CRRT, at 1.1%).
# Any eICU window with crrt=1 therefore gets log P = -inf under three states and
# the whole patient sequence collapses or goes NaN. Flooring the categorical
# emissions is a transport-time numerical guard, pre-specified here rather than
# patched in when the decode crashes. 06_verify_epsilon_floor.py checks it does
# not change MIMIC test-set decoding.
EPS = 1e-4

# ---------------------------------------------------------------------------
# 1. imputation medians -- replicate 03_impute_missing.py exactly
# ---------------------------------------------------------------------------
VITAL_TIER = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c",
              "gcs_eye", "gcs_verbal", "gcs_motor", "urine_output_ml"]
CBC_PANEL = ["wbc", "hemoglobin", "platelet"]
CHEM_PANEL = ["creatinine", "bun", "sodium", "potassium", "glucose"]
LAB_TIER = CBC_PANEL + CHEM_PANEL
FFILL_LIMIT = {**{v: 1 for v in VITAL_TIER}, **{v: 2 for v in LAB_TIER}}

cohort = pd.read_csv(f"{T1}/patient_level_cohort.csv")
split = pd.read_csv(f"{T1}/patient_train_test_split.csv")
wide = pd.read_csv(f"{T1}/timewindow_level_raw.csv")
wide = wide.merge(cohort[["stay_id", "subject_id"]], on="stay_id", how="left")
wide = wide.merge(split[["subject_id", "split"]], on="subject_id", how="left")
wide = wide.sort_values(["stay_id", "window_idx"])

rows = []
for var in VITAL_TIER + LAB_TIER:
    filled = wide.groupby("stay_id")[var].ffill(limit=FFILL_LIMIT[var])
    med = filled[wide.split == "train"].median()
    wide[var] = filled.fillna(med)          # keep Paper 1's sequential ordering
    rows.append({"variable": var,
                 "tier": "vital" if var in VITAL_TIER else "lab",
                 "ffill_limit_windows": FFILL_LIMIT[var],
                 "train_median_post_ffill": med})
imp = pd.DataFrame(rows)
imp.to_csv(f"{OUT}/imputation_medians_train.csv", index=False)
imp[["variable", "ffill_limit_windows"]].to_csv(f"{OUT}/ffill_limits.csv", index=False)

# cross-check against the Paper 1 imputed file: the imputed values we just
# reproduced must equal the ones actually used in the manuscript
ref = pd.read_csv(f"{T1}/timewindow_level_imputed.csv").sort_values(["stay_id", "window_idx"])
chk = []
for var in VITAL_TIER + LAB_TIER:
    a = wide[var].to_numpy(); b = ref[var].to_numpy()
    chk.append({"variable": var, "max_abs_diff_vs_paper1": float(np.nanmax(np.abs(a - b)))})
chk = pd.DataFrame(chk)
worst = chk.max_abs_diff_vs_paper1.max()
print(f"Imputation replay check -- max abs difference vs Paper 1 imputed file: {worst:.3g}")
if worst > 1e-9:
    print(chk.sort_values("max_abs_diff_vs_paper1", ascending=False).to_string(index=False))
    raise SystemExit("REPLAY MISMATCH: frozen medians would not reproduce Paper 1")

# ---------------------------------------------------------------------------
# 2. z-score parameters (already persisted by Paper 1; copied so the frozen set
#    is self-contained and cannot drift if 04_outputs is ever regenerated)
# ---------------------------------------------------------------------------
z = pd.read_csv(f"{T1}/standardization_params.csv", index_col=0)
z.index.name = "variable"
z.to_csv(f"{OUT}/zscore_params_train.csv")

# ---------------------------------------------------------------------------
# 3. model parameters
# ---------------------------------------------------------------------------
model = torch.load(f"{ROOT}/.model_k4.pt", weights_only=False)
CONT_Z, n_cont, _ = feature_spec.features(pd.read_csv(f"{T1}/timewindow_level_modeling.csv", nrows=1).columns,
                                          include_treatment=False)
BINARY = feature_spec.TREATMENT_BINARY
FEATURES = CONT_Z + BINARY
pd.DataFrame({"position": range(len(FEATURES)), "feature": FEATURES,
              "kind": ["continuous_z"] * len(CONT_Z) + ["binary"] * len(BINARY)}
             ).to_csv(f"{OUT}/feature_order.csv", index=False)
print(f"Feature order frozen: {len(CONT_Z)} continuous + {len(BINARY)} binary = {len(FEATURES)}")

lbl = pd.read_csv(f"{T1}/state_label_mapping.csv").set_index("raw_state")

brows = []
for raw_state, dist in enumerate(model.distributions):
    subs = dist.distributions[-len(BINARY):]
    for name, sub in zip(BINARY, subs):
        p1 = float(sub.probs.detach().numpy().ravel()[1])
        brows.append({"raw_state": raw_state,
                      "canonical_state": int(lbl.loc[raw_state, "canonical_state"]),
                      "state_name": lbl.loc[raw_state, "name"],
                      "variable": name,
                      "p1_fitted": p1,
                      "p1_floored": float(np.clip(p1, EPS, 1 - EPS))})
bdf = pd.DataFrame(brows)
bdf.to_csv(f"{OUT}/emission_binary_frozen.csv", index=False)
n_deg = int((bdf.p1_fitted <= 0).sum() + (bdf.p1_fitted >= 1).sum())
print(f"Binary emissions: {n_deg} of {len(bdf)} are degenerate (p=0 or 1) and required the {EPS} floor")
print(bdf[(bdf.p1_fitted <= 0) | (bdf.p1_fitted >= 1)][["state_name", "variable", "p1_fitted"]].to_string(index=False))

trans = torch.exp(model.edges).detach().numpy() if model.edges.min() < 0 else model.edges.detach().numpy()
starts = torch.exp(model.starts).detach().numpy() if model.starts.min() < 0 else model.starts.detach().numpy()
names = [lbl.loc[i, "name"] for i in range(len(model.distributions))]
# pomegranate's DenseHMM keeps an implicit end state, so the rows of `edges` sum
# to less than 1; the remainder is P(sequence ends in this state). It is carried
# here as its own column so the frozen matrix is complete. NOTE for Paper 2: the
# MIMIC-vs-eICU transition comparison must use the row-normalised EMPIRICAL
# matrix (04_outputs/tables/transition_matrix_empirical_k4.csv), which is what
# the manuscript reports -- not this one.
ends = torch.exp(model.ends).detach().numpy() if model.ends.min() < 0 else model.ends.detach().numpy()
tdf = pd.DataFrame(trans, index=names, columns=names)
tdf["_end"] = ends
tdf.to_csv(f"{OUT}/transition_frozen.csv")
pd.DataFrame({"state_name": names, "start_prob": starts}).to_csv(f"{OUT}/start_frozen.csv", index=False)

print(f"\nWrote frozen parameter set to {OUT}")
for f in ["imputation_medians_train.csv", "ffill_limits.csv", "zscore_params_train.csv",
          "emission_binary_frozen.csv", "transition_frozen.csv", "start_frozen.csv",
          "feature_order.csv"]:
    print(f"  {f}")
