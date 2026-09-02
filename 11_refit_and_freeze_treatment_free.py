"""
Paper 2, step 9a: reproduce and freeze the MIMIC treatment-free model.

Protocol v1.1 section 10.3 makes the 17-variable treatment-free model a
co-reported specificity control. Paper 1 fitted it (03_code/11_sensitivity_no_
treatment_vars.py) but **never saved the model object** -- only the state
assignments, profile and crosstab CSVs. It therefore has to be refitted before
it can be transported.

A refit is only admissible if it reproduces the published assignments exactly.
`fit_best_of_n` seeds both KMeans and the HMM from the restart index, so the
procedure is deterministic given the same input; this script asserts the
reproduction and refuses to freeze anything if it fails. Same discipline as the
epsilon-floor verification.

State naming: the treatment-free states are a different clustering, not the same
states with a variable removed. They are matched to the four primary states by
standardized profile correlation and named "<primary state> analogue", so no
result can silently imply they are identical objects.

Outputs (frozen_params_treatment_free/):
  feature_order.csv, transition_frozen.csv, start_frozen.csv,
  emission_continuous_frozen.csv, state_label_mapping.csv,
  mimic_reference_profile.csv, reproduction_check.csv
"""
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "03_code"))
import feature_spec
from hmm_fit_utils import fit_best_of_n
from pomegranate_patches import CategoricalMixed          # noqa: F401

HERE = Path(__file__).parent
OUT = HERE / "frozen_params_treatment_free"; OUT.mkdir(exist_ok=True)
T1 = ROOT / "04_outputs/tables"

wide = pd.read_csv(T1 / "timewindow_level_modeling.csv")
train = wide[wide.split == "train"].sort_values(["stay_id", "window_idx"]).reset_index(drop=True)
CONT_Z, N_CONT, _ = feature_spec.features(wide.columns, include_treatment=False)
print(f"Treatment-free feature set: {N_CONT} continuous\n  {CONT_Z}")

seqs = [torch.tensor(g[CONT_Z].values, dtype=torch.float32)
        for _, g in train.groupby("stay_id", sort=False)]
print(f"Refitting on {len(seqs):,} MIMIC training patients, "
      f"{sum(len(s) for s in seqs):,} windows, 10 restarts ...")
res = fit_best_of_n(seqs, K=4, n_cont=N_CONT, n_bin=0, n_restarts=10, verbose=True)
model, labels = res["model"], res["labels"]
print(f"\nBIC={res['bic']:.1f}  restart-stability ARI={res['stability_ari']:.3f}")

# ---------------------------------------------------------------------------
# reproduction check against the published assignments
# ---------------------------------------------------------------------------
pub = pd.read_csv(T1 / "sensitivity_notreat_state_assignments.csv")
mine = train[["stay_id", "window_idx"]].copy(); mine["refit"] = labels
chk = pub.merge(mine, on=["stay_id", "window_idx"], how="inner")
exact = float((chk.state_no_treatment == chk.refit).mean())
ari = adjusted_rand_score(chk.state_no_treatment, chk.refit)
summ = pd.read_csv(T1 / "sensitivity_notreat_summary.csv").iloc[0]
print(f"\nReproduction vs published assignments: {len(chk):,} windows matched")
print(f"  exact label agreement : {exact*100:.4f}%")
print(f"  ARI                   : {ari:.6f}")
print(f"  published BIC {summ.BIC:.1f} vs refit {res['bic']:.1f}; "
      f"published stability ARI {summ.restart_stability_ARI:.3f} vs refit {res['stability_ari']:.3f}")
pd.DataFrame([{"windows_compared": len(chk), "exact_label_agreement": exact, "ari": ari,
               "published_bic": float(summ.BIC), "refit_bic": float(res["bic"]),
               "published_stability_ari": float(summ.restart_stability_ARI),
               "refit_stability_ari": float(res["stability_ari"])}]) \
    .to_csv(OUT / "reproduction_check.csv", index=False)
if ari < 0.999:
    raise SystemExit("REFIT DID NOT REPRODUCE the published treatment-free model -- "
                     "do not freeze; investigate before transporting.")
print("  -> reproduction OK, freezing")

# ---------------------------------------------------------------------------
# name the states by matching profiles to the primary model
# ---------------------------------------------------------------------------
prim_lbl = pd.read_csv(T1 / "state_label_mapping.csv").set_index("raw_state")
prim = torch.load(ROOT / ".model_k4.pt", weights_only=False)
PRIM_CONT = [c for c in pd.read_csv(HERE / "frozen_params/feature_order.csv").feature if c.endswith("_z")]

def profile(m, feats):
    return np.array([[float(d.distributions[i].means.detach().ravel()[0])
                      for i in range(len(feats))] for d in m.distributions])
P, Q = profile(prim, PRIM_CONT), profile(model, CONT_Z)
assert PRIM_CONT == CONT_Z, "feature order differs between the two models"

corr = np.array([[np.corrcoef(Q[j], P[k])[0, 1] for k in range(4)] for j in range(4)])
# greedy best match, highest correlation first, one-to-one
pairs, used = {}, set()
for j, k in sorted(((j, k) for j in range(4) for k in range(4)),
                   key=lambda t: -corr[t[0], t[1]]):
    if j not in pairs and k not in used:
        pairs[j], used_ = k, used.add(k)
mapping = pd.DataFrame([{"tf_state": j, "matched_primary_raw_state": pairs[j],
                         "name": prim_lbl.loc[pairs[j], "name"] + " analogue",
                         "profile_correlation": round(float(corr[j, pairs[j]]), 4)}
                        for j in range(4)]).sort_values("tf_state")
mapping.to_csv(OUT / "state_label_mapping.csv", index=False)
print("\nTreatment-free states matched to primary states by profile correlation:")
print(mapping.to_string(index=False))

# ---------------------------------------------------------------------------
# freeze parameters
# ---------------------------------------------------------------------------
pd.DataFrame({"position": range(len(CONT_Z)), "feature": CONT_Z,
              "kind": "continuous_z"}).to_csv(OUT / "feature_order.csv", index=False)
rows = []
for j, d in enumerate(model.distributions):
    for i, f in enumerate(CONT_Z):
        sub = d.distributions[i]
        rows.append({"tf_state": j, "name": mapping.set_index("tf_state").loc[j, "name"],
                     "feature": f, "mean_z": float(sub.means.detach().ravel()[0]),
                     "sd_z": float(torch.sqrt(sub.covs.detach().ravel())[0])})
pd.DataFrame(rows).to_csv(OUT / "emission_continuous_frozen.csv", index=False)

names = [mapping.set_index("tf_state").loc[j, "name"] for j in range(4)]
tr = torch.exp(model.edges).detach().numpy() if model.edges.min() < 0 else model.edges.detach().numpy()
en = torch.exp(model.ends).detach().numpy() if model.ends.min() < 0 else model.ends.detach().numpy()
st = torch.exp(model.starts).detach().numpy() if model.starts.min() < 0 else model.starts.detach().numpy()
t = pd.DataFrame(tr, index=names, columns=names); t["_end"] = en
t.to_csv(OUT / "transition_frozen.csv")
pd.DataFrame({"state_name": names, "start_prob": st}).to_csv(OUT / "start_frozen.csv", index=False)
torch.save(model, HERE / "frozen_params_treatment_free/.model_k4_treatment_free.pt")

# ---------------------------------------------------------------------------
# MIMIC reference: decode ALL patients, build the comparison target
# ---------------------------------------------------------------------------
allw = wide.sort_values(["stay_id", "window_idx"])
lab = [model.predict(torch.tensor(g[CONT_Z].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
       for _, g in allw.groupby("stay_id", sort=False)]
allw = allw.copy(); allw["tf_state"] = np.concatenate(lab)
allw["state"] = allw.tf_state.map(mapping.set_index("tf_state")["name"])

raw_cols = ["gcs_eye", "gcs_motor", "heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c",
            "wbc", "hemoglobin", "platelet", "creatinine", "bun", "sodium", "potassium",
            "glucose", "urine_output_ml"]
prof = allw.groupby("state").agg(window_share_pct=("tf_state", lambda s: round(100*len(s)/len(allw), 2)))
for c in raw_cols:
    prof[c] = allw.groupby("state")[c].median().round(2)
prof.to_csv(OUT / "mimic_reference_profile.csv")
print("\nMIMIC treatment-free reference, window share:")
print(prof[["window_share_pct"]].to_string())

# transitions and dynamics, the pre-registered comparison targets
seq = allw[["stay_id", "window_idx", "state"]].copy()
seq["next"] = seq.groupby("stay_id")["state"].shift(-1)
tm = pd.crosstab(seq.state, seq["next"], normalize="index").round(4)
tm.to_csv(OUT / "mimic_reference_transitions.csv")
changed = seq.groupby("stay_id")["state"].nunique().gt(1).mean()
print(f"\nMIMIC treatment-free: {changed*100:.1f}% of patients changed state at least once "
      f"(primary model: 40.3%)")
print(f"self-transition probabilities: "
      + ", ".join(f"{s.split(' analogue')[0][:22]} {tm.loc[s, s]:.3f}" for s in tm.index))
print(f"\nWrote {OUT}/")
