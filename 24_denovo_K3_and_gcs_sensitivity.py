"""
Paper 2, revisions requested by the PI, 23 August 2026.

(A) The de novo K=3 solution. eICU's own most stable solution is three states,
    not four, and the manuscript should not imply otherwise. If those three
    correspond to the three MIMIC states that succeed everywhere else, the
    argument closes: eICU supports three robust states and the fourth MIMIC
    state is a residual that does not re-emerge.

(B) Observed-GCS versus imputed-GCS sensitivity for the FULL model. The single
    most attackable point in the paper is that transport success might be
    mechanically inflated by imputation, because the discovery imputation median
    for gcs_motor coincides exactly with the preserved state's point-mass
    emission. Two questions, both answerable without refitting:
      1. Is preserved-state enrichment concentrated in imputed-GCS windows?
      2. Do the state profile correlations hold when computed on observed-GCS
         windows only?
"""
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "03_code"))
from hmm_fit_utils import fit_best_of_n
from pomegranate_patches import CategoricalMixed          # noqa: F401

HERE = Path(__file__).parent
OUT_D = HERE / "results_denovo"; OUT_F = HERE / "results_full_model"
FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
CONT = [f for f in FEATS if f.endswith("_z")]
RAW = [f[:-2] for f in CONT]
BIN = [f for f in FEATS if not f.endswith("_z")]
lblmap = pd.read_csv(ROOT / "04_outputs/tables/state_label_mapping.csv").set_index("raw_state")["name"].to_dict()

# =========================================================================
# (A) de novo K=3
# =========================================================================
cont = pd.read_csv(HERE / "windows/timewindow_level_raw_eicu.csv")
bina = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv")
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
elig = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
keep = elig & set(coh.loc[coh.vent_ascertainable == 1, "patientunitstayid"])
w = cont.merge(bina, on=["patientunitstayid", "window_idx"])
w = w[w.patientunitstayid.isin(keep)].merge(coh[["patientunitstayid", "stroke_subtype"]],
                                            on="patientunitstayid")
pts = w[["patientunitstayid", "stroke_subtype"]].drop_duplicates()
tr, _ = train_test_split(pts, test_size=0.30, random_state=42, stratify=pts.stroke_subtype)
w["split"] = np.where(w.patientunitstayid.isin(set(tr.patientunitstayid)), "train", "test")
LIM = {**{v: 1 for v in ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c",
                         "gcs_eye", "gcs_motor", "urine_output_ml"]},
       **{v: 2 for v in ["wbc", "hemoglobin", "platelet", "creatinine", "bun",
                         "sodium", "potassium", "glucose"]}}
w = w.sort_values(["patientunitstayid", "window_idx"]).reset_index(drop=True)
for v in RAW:
    w[v] = w.groupby("patientunitstayid")[v].ffill(limit=LIM[v])
    w[v] = w[v].fillna(w.loc[w.split == "train", v].median())
    mu, sd = w.loc[w.split == "train", v].mean(), w.loc[w.split == "train", v].std()
    w[f"{v}_z"] = (w[v] - mu) / sd
seqs = [torch.tensor(g[FEATS].to_numpy(dtype=np.float32))
        for _, g in w[w.split == "train"].groupby("patientunitstayid", sort=False)]
print(f"Refitting de novo K=3 on {len(seqs):,} eICU training patients ...")
r3 = fit_best_of_n(seqs, K=3, n_cont=len(CONT), n_bin=len(BIN), n_restarts=5, verbose=False)
m3 = r3["model"]
mimic = torch.load(ROOT / ".model_k4.pt", weights_only=False)
def prof(m, n): return np.array([[float(d.distributions[i].means.detach().ravel()[0])
                                  for i in range(n)] for d in m.distributions])
P, Q = prof(mimic, len(CONT)), prof(m3, len(CONT))
corr = np.array([[np.corrcoef(Q[j], P[k])[0, 1] for k in range(4)] for j in range(3)])
pairs, used = {}, set()
for j, k in sorted(((j, k) for j in range(3) for k in range(4)), key=lambda t: -corr[t[0], t[1]]):
    if j not in pairs and k not in used:
        pairs[j] = k; used.add(k)
FLOOR = 0.60
k3 = pd.DataFrame([{"eicu_state": j,
                    "best_mimic_candidate": lblmap[pairs[j]],
                    "profile_correlation": round(float(corr[j, pairs[j]]), 3),
                    "corresponds": bool(corr[j, pairs[j]] >= FLOOR)} for j in range(3)])
lab3 = np.concatenate([m3.predict(torch.tensor(g[FEATS].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
                       for _, g in w.groupby("patientunitstayid", sort=False)])
w["k3"] = lab3
k3["pct_windows"] = [round(float((w.k3 == j).mean() * 100), 1) for j in range(3)]
unmatched = set(range(4)) - set(pairs.values())
k3.to_csv(OUT_D / "denovo_K3_matching.csv", index=False)
torch.save(m3, OUT_D / ".model_k3_eicu_denovo.pt")
print("\n=== de novo K=3, matched to MIMIC states by profile only ===")
print(k3.to_string(index=False))
print(f"MIMIC state with no K=3 counterpart: {lblmap[list(unmatched)[0]]}")
print(f"restart stability ARI: {r3['stability_ari']:.3f}  (K=4 was 0.840)")

# =========================================================================
# (B) observed vs imputed GCS, full model
# =========================================================================
a = pd.read_csv(OUT_F / "eicu_state_assignments_full_A1.csv")
mod = pd.read_csv(HERE / "windows/timewindow_level_modeling_eicu_A1.csv")
b2 = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv")
d = mod.merge(b2, on=["patientunitstayid", "window_idx"]).merge(
    a[["patientunitstayid", "window_idx", "state", "vent_ascertainable"]],
    on=["patientunitstayid", "window_idx"])
d["gcs_observed"] = (d.gcs_motor_missing_raw == 0)

mw = pd.read_csv(ROOT / "04_outputs/tables/timewindow_level_modeling.csv").sort_values(["stay_id", "window_idx"])
floored = torch.load(ROOT / ".model_k4.pt", weights_only=False)
for dd in floored.distributions:
    for sub in dd.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), 1e-4, 1 - 1e-4); p = p / p.sum(-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False); sub._log_probs = torch.log(p)
mw["state"] = pd.Series(np.concatenate(
    [floored.predict(torch.tensor(g[FEATS].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
     for _, g in mw.groupby("stay_id", sort=False)]), index=mw.index).map(lblmap)
mw["gcs_observed"] = (mw.gcs_motor_missing_raw == 0)
STATES = sorted(mw.state.unique())
MZ_all = mw.groupby("state")[FEATS].mean()
MZ_obs = mw[mw.gcs_observed].groupby("state")[FEATS].mean()

rows = []
for scope, sub in [("full", d), ("both-eligible", d[d.patientunitstayid.isin(elig) & (d.vent_ascertainable == 1)])]:
    for s in STATES:
        o = sub[sub.gcs_observed]; i = sub[~sub.gcs_observed]
        rows.append({"scope": scope, "state": s,
                     "mimic_pct": round(float((mw.state == s).mean() * 100), 1),
                     "eicu_all_pct": round(float((sub.state == s).mean() * 100), 1),
                     "eicu_observed_gcs_pct": round(float((o.state == s).mean() * 100), 1),
                     "eicu_imputed_gcs_pct": round(float((i.state == s).mean() * 100), 1),
                     "mimic_observed_gcs_pct": round(float((mw[mw.gcs_observed].state == s).mean() * 100), 1)})
split = pd.DataFrame(rows); split.to_csv(OUT_F / "gcs_observed_vs_imputed_prevalence.csv", index=False)
print("\n=== state prevalence, observed vs imputed GCS (full model) ===")
for sc in ["full", "both-eligible"]:
    t = split[split.scope == sc]
    print(f"\n-- {sc} --")
    print(t[["state", "mimic_pct", "eicu_all_pct", "eicu_observed_gcs_pct", "eicu_imputed_gcs_pct"]].to_string(index=False))

# profile correlations restricted to observed-GCS windows, both sides
cr = []
for scope, sub in [("full", d), ("both-eligible", d[d.patientunitstayid.isin(elig) & (d.vent_ascertainable == 1)])]:
    ez_all = sub.groupby("state")[FEATS].mean().reindex(STATES)
    ez_obs = sub[sub.gcs_observed].groupby("state")[FEATS].mean().reindex(STATES)
    for s in STATES:
        cr.append({"scope": scope, "state": s,
                   "corr_all_windows": round(float(np.corrcoef(MZ_all.loc[s], ez_all.loc[s])[0, 1]), 3),
                   "corr_observed_gcs_only": round(float(np.corrcoef(MZ_obs.loc[s], ez_obs.loc[s])[0, 1]), 3)})
cc = pd.DataFrame(cr); cc.to_csv(OUT_F / "gcs_observed_only_profile_corr.csv", index=False)
print("\n=== profile correlation, all windows vs observed-GCS windows only ===")
print("(MIMIC side restricted to observed-GCS windows too, so the comparison is like for like)")
for sc in ["full", "both-eligible"]:
    t = cc[cc.scope == sc]
    print(f"\n-- {sc} --  min all {t.corr_all_windows.min():.3f} -> min observed-only {t.corr_observed_gcs_only.min():.3f}")
    print(t[["state", "corr_all_windows", "corr_observed_gcs_only"]].to_string(index=False))
