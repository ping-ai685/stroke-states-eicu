"""
Paper 2, Stage D: de novo eICU HMM, structural reproducibility.

Protocol section 13. Runs only now that the frozen-model analysis is complete and
archived. This asks whether eICU independently organises into a similar four-state
structure, not whether the MIMIC model can assign eICU observations.

  - harmonised 21-variable feature set, same 6-hour representation
  - patient-level train/test split inside eICU, all windows of a patient together
  - preprocessing estimated in the eICU TRAIN split only
  - multiple restarts, convergence and stability reported
  - K = 3-6 with the discovery selection criteria
  - eICU states matched to MIMIC states by feature profile only, no outcome data;
    matching rule fixed before any outcome is looked at

If K=4 is unstable or the structure differs materially, that is reported. Clinical
names are not forced onto non-corresponding states.
"""
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "03_code"))
from hmm_fit_utils import fit_best_of_n
from pomegranate_patches import CategoricalMixed          # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "results_denovo"; OUT.mkdir(exist_ok=True)
FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
CONT = [f for f in FEATS if f.endswith("_z")]
RAW = [f[:-2] for f in CONT]
BIN = [f for f in FEATS if not f.endswith("_z")]

cont = pd.read_csv(HERE / "windows/timewindow_level_raw_eicu.csv")
bina = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv")
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
elig = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
keep = elig & set(coh.loc[coh.vent_ascertainable == 1, "patientunitstayid"])
w = cont.merge(bina, on=["patientunitstayid", "window_idx"])
w = w[w.patientunitstayid.isin(keep)].merge(
    coh[["patientunitstayid", "stroke_subtype"]], on="patientunitstayid")
print(f"de novo cohort (GCS-eligible AND vent-ascertainable): "
      f"{w.patientunitstayid.nunique():,} patients, {len(w):,} windows")

# ---- eICU-internal split and preprocessing ---------------------------------
pts = w[["patientunitstayid", "stroke_subtype"]].drop_duplicates()
tr, te = train_test_split(pts, test_size=0.30, random_state=42, stratify=pts.stroke_subtype)
w["split"] = np.where(w.patientunitstayid.isin(set(tr.patientunitstayid)), "train", "test")
LIM = {**{v: 1 for v in ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c",
                         "gcs_eye", "gcs_motor", "urine_output_ml"]},
       **{v: 2 for v in ["wbc", "hemoglobin", "platelet", "creatinine", "bun",
                         "sodium", "potassium", "glucose"]}}
w = w.sort_values(["patientunitstayid", "window_idx"]).reset_index(drop=True)
params = {}
for v in RAW:
    w[v] = w.groupby("patientunitstayid")[v].ffill(limit=LIM[v])
    med = w.loc[w.split == "train", v].median()
    w[v] = w[v].fillna(med)
    mu, sd = w.loc[w.split == "train", v].mean(), w.loc[w.split == "train", v].std()
    w[f"{v}_z"] = (w[v] - mu) / sd
    params[v] = {"train_median": med, "train_mean": mu, "train_sd": sd}
pd.DataFrame(params).T.to_csv(OUT / "eicu_train_preprocessing_params.csv")

train = w[w.split == "train"]
seqs = [torch.tensor(g[FEATS].to_numpy(dtype=np.float32))
        for _, g in train.groupby("patientunitstayid", sort=False)]
print(f"eICU train: {len(seqs):,} patients, {sum(len(s) for s in seqs):,} windows")

# ---- K selection ------------------------------------------------------------
res, models = [], {}
for K in [3, 4, 5, 6]:
    r = fit_best_of_n(seqs, K=K, n_cont=len(CONT), n_bin=len(BIN), n_restarts=5, verbose=False)
    if r is None:
        res.append({"K": K, "BIC": np.nan, "restart_stability_ARI": np.nan,
                    "min_state_share_pct": np.nan, "note": "all restarts failed"}); continue
    models[K] = r
    share = pd.Series(r["labels"]).value_counts(normalize=True) * 100
    res.append({"K": K, "BIC": round(r["bic"], 1),
                "restart_stability_ARI": round(r["stability_ari"], 3),
                "min_state_share_pct": round(float(share.min()), 1),
                "n_states_recovered": int(share.size), "note": ""})
    print(f"  K={K}: BIC {r['bic']:.0f}, stability ARI {r['stability_ari']:.3f}, "
          f"min share {share.min():.1f}%")
ks = pd.DataFrame(res); ks.to_csv(OUT / "denovo_K_selection.csv", index=False)
print("\n=== K selection ===")
print(ks.to_string(index=False))

if 4 not in models:
    raise SystemExit("K=4 did not converge in eICU -- report as a structural finding")

# ---- match K=4 de novo states to MIMIC states, profiles only ----------------
m4 = models[4]["model"]
lblmap = pd.read_csv(ROOT / "04_outputs/tables/state_label_mapping.csv").set_index("raw_state")["name"].to_dict()
mimic = torch.load(ROOT / ".model_k4.pt", weights_only=False)
def prof(model):
    return np.array([[float(d.distributions[i].means.detach().ravel()[0]) for i in range(len(CONT))]
                     for d in model.distributions])
P, Q = prof(mimic), prof(m4)
corr = np.array([[np.corrcoef(Q[j], P[k])[0, 1] for k in range(4)] for j in range(4)])
pairs, used = {}, set()
for j, k in sorted(((j, k) for j in range(4) for k in range(4)), key=lambda t: -corr[t[0], t[1]]):
    if j not in pairs and k not in used:
        pairs[j] = k; used.add(k)
# Protocol section 13 forbids forcing clinical names onto non-corresponding
# states. Greedy matching assigns all four regardless of how poor the best match
# is, so a correspondence floor is applied: a de novo state whose best profile
# correlation falls below the section 11.2 falsification floor of 0.60 is left
# UNMATCHED and keeps no MIMIC name.
MATCH_FLOOR = 0.60
mp = pd.DataFrame([{"eicu_state": j,
                    "matched_mimic_state": (lblmap[pairs[j]] if corr[j, pairs[j]] >= MATCH_FLOOR
                                            else f"eICU state {j} (UNMATCHED)"),
                    "best_mimic_candidate": lblmap[pairs[j]],
                    "profile_correlation": round(float(corr[j, pairs[j]]), 3),
                    "corresponds": bool(corr[j, pairs[j]] >= MATCH_FLOOR)} for j in range(4)])
mp.to_csv(OUT / "denovo_state_matching.csv", index=False)
print("\n=== de novo eICU states matched to MIMIC states (profiles only, no outcome data) ===")
print(mp.to_string(index=False))

# ---- ARI against the frozen A1 assignment -----------------------------------
allseq = [(sid, torch.tensor(g[FEATS].to_numpy(dtype=np.float32)))
          for sid, g in w.groupby("patientunitstayid", sort=False)]
lab = np.concatenate([m4.predict(s.unsqueeze(0))[0].numpy() for _, s in allseq])
w["denovo_state"] = lab
w["denovo_name"] = w.denovo_state.map(mp.set_index("eicu_state")["matched_mimic_state"])
froz = pd.read_csv(HERE / "results_full_model/eicu_state_assignments_full_A1.csv")
cmp = w[["patientunitstayid", "window_idx", "denovo_name"]].merge(
    froz[["patientunitstayid", "window_idx", "state"]], on=["patientunitstayid", "window_idx"])
ari = adjusted_rand_score(cmp.state, cmp.denovo_name)
agree = float((cmp.state == cmp.denovo_name).mean())
print(f"\nde novo vs frozen A1 on the same windows: ARI {ari:.3f}, label agreement {agree*100:.1f}% "
      f"({len(cmp):,} windows)")
prev = pd.DataFrame({
    "denovo_pct": (w.denovo_name.value_counts(normalize=True) * 100).round(1),
    "frozen_A1_pct": (froz[froz.patientunitstayid.isin(keep)].state.value_counts(normalize=True) * 100).round(1),
    "mimic_pct": pd.Series({"neurologically preserved-low support": 63.3,
                            "neurological impairment-respiratory support": 17.1,
                            "neurological impairment-renal dysfunction": 11.8,
                            "neurological impairment-low support": 7.8})})
prev.to_csv(OUT / "denovo_prevalence.csv")
print("\n=== state prevalence ===")
print(prev.to_string())
w[["patientunitstayid", "window_idx", "denovo_state", "denovo_name", "split"]].to_csv(
    OUT / "eicu_denovo_assignments.csv", index=False)
torch.save(m4, OUT / ".model_k4_eicu_denovo.pt")
pd.DataFrame([{"ari_vs_frozen_A1": round(ari, 3), "label_agreement": round(agree, 3),
               "n_windows": len(cmp)}]).to_csv(OUT / "denovo_vs_frozen.csv", index=False)
