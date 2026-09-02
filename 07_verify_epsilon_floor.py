"""
Paper 2, step 8 (protocol v1.1 section 9.1): verify that the categorical-emission
floor leaves MIMIC decoding unchanged before it is applied to eICU.

Why the floor exists. In the fitted K=4 model, P(crrt = 1) is 0.0113 in the
renal-dysfunction state and 0.0, 1.5e-14 and 7.2e-11 in the other three. A
window with crrt = 1 therefore gets zero or near-zero likelihood under three of
four states. In MIMIC that is harmless -- CRRT occurs in 0.2% of windows and
almost all of them are already in the renal state -- but in eICU, where the CRRT
flag is rebuilt from a different source with a different error profile, a single
misassigned CRRT window can drive an entire patient sequence into one state or
produce NaN.

The floor is a transport-time numerical guard, not a change to the published
model. This script proves that claim on MIMIC: it must show identical decoding.
If it does not, the floor is not admissible and the protocol needs a different
remedy.

Outputs (audit/):
  epsilon_floor_verification.csv   the decision table
"""
import copy
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "03_code"))
import feature_spec
from pomegranate_patches import CategoricalMixed          # noqa: F401  (needed to unpickle)

OUT = Path(__file__).parent / "audit"
EPS = 1e-4

wide = pd.read_csv(ROOT / "04_outputs/tables/timewindow_level_modeling.csv")
FEATURES = pd.read_csv(Path(__file__).parent / "frozen_params/feature_order.csv").feature.tolist()
BINARY = feature_spec.TREATMENT_BINARY
lbl = pd.read_csv(ROOT / "04_outputs/tables/state_label_mapping.csv").set_index("raw_state")

model = torch.load(ROOT / ".model_k4.pt", weights_only=False)

# ---------------------------------------------------------------------------
# 1. build the floored copy
# ---------------------------------------------------------------------------
floored = copy.deepcopy(model)
changes = []
for k, dist in enumerate(floored.distributions):
    for name, sub in zip(BINARY, dist.distributions[-len(BINARY):]):
        p = sub.probs.detach().clone()
        before = float(p[0, 1])
        p = torch.clamp(p, EPS, 1 - EPS)
        p = p / p.sum(dim=-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False) \
            if isinstance(sub.probs, torch.nn.Parameter) else p
        sub._log_probs = torch.log(p)
        after = float(p[0, 1])
        if abs(after - before) > 1e-12:
            changes.append({"state": lbl.loc[k, "name"], "variable": name,
                            "p1_before": before, "p1_after": after})
ch = pd.DataFrame(changes)
print(f"Emission probabilities altered by the floor: {len(ch)} of {4*len(BINARY)}")
print(ch.to_string(index=False) if len(ch) else "  (none)")

# ---------------------------------------------------------------------------
# 2. demonstrate the failure mode the floor removes
# ---------------------------------------------------------------------------
row = wide.sort_values(["stay_id", "window_idx"]).iloc[0][FEATURES].astype(float).to_numpy()
probe = row.copy()
probe[FEATURES.index("crrt")] = 1.0
x = torch.tensor(probe, dtype=torch.float32).reshape(1, 1, -1)
print("\nPer-state log-probability of one window with crrt=1:")
print(f"  {'state':<44}{'original':>12}{'floored':>12}")
for k in range(len(model.distributions)):
    a = float(model.distributions[k].log_probability(x[0]))
    b = float(floored.distributions[k].log_probability(x[0]))
    flag = "  <-- -inf under the published model" if np.isinf(a) else ""
    print(f"  {lbl.loc[k,'name']:<44}{a:>12.2f}{b:>12.2f}{flag}")

# ---------------------------------------------------------------------------
# 3. decode every MIMIC patient under both models
# ---------------------------------------------------------------------------
data = wide.sort_values(["stay_id", "window_idx"])
seqs, ids = [], []
for stay_id, g in data.groupby("stay_id", sort=False):
    seqs.append(torch.tensor(g[FEATURES].to_numpy(dtype=np.float32)))
    ids.append(stay_id)
print(f"\nDecoding {len(seqs):,} patients, {sum(len(s) for s in seqs):,} windows, under both models ...")

def decode(m):
    return [m.predict(s.unsqueeze(0))[0].numpy() for s in seqs]
a_lab, b_lab = decode(model), decode(floored)
A, Bf = np.concatenate(a_lab), np.concatenate(b_lab)

agree = float((A == Bf).mean())
ari = adjusted_rand_score(A, Bf)
n_pat_changed = sum(1 for x_, y_ in zip(a_lab, b_lab) if not np.array_equal(x_, y_))
print(f"\nWindow-level agreement : {agree*100:.6f}%  ({int((A != Bf).sum())} of {len(A):,} windows differ)")
print(f"Adjusted Rand index    : {ari:.6f}")
print(f"Patients with any change: {n_pat_changed} of {len(seqs):,}")
print(f"Non-finite log-prob under the published model: "
      f"{sum(1 for s in seqs if not np.isfinite(model.log_probability(s.unsqueeze(0)).item()))}")

prev = pd.DataFrame({
    "state": [lbl.loc[k, "name"] for k in range(4)],
    "pct_windows_original": [round((A == k).mean() * 100, 4) for k in range(4)],
    "pct_windows_floored": [round((Bf == k).mean() * 100, 4) for k in range(4)],
})
prev["difference"] = (prev.pct_windows_floored - prev.pct_windows_original).round(6)
print("\n" + prev.to_string(index=False))

# CRRT windows specifically -- the ones the floor could plausibly move
crrt_mask = data.crrt.to_numpy() == 1
print(f"\nWindows with crrt=1: {int(crrt_mask.sum()):,} ({crrt_mask.mean()*100:.2f}%); "
      f"of these, {int((A[crrt_mask] != Bf[crrt_mask]).sum())} changed state")

verdict = "ADMISSIBLE" if (agree == 1.0) else "NOT ADMISSIBLE -- decoding changed"
pd.DataFrame([
    {"check": "emission probabilities altered", "value": len(ch)},
    {"check": "windows decoded", "value": len(A)},
    {"check": "windows differing", "value": int((A != Bf).sum())},
    {"check": "window-level agreement", "value": agree},
    {"check": "adjusted Rand index", "value": ari},
    {"check": "patients with any change", "value": n_pat_changed},
    {"check": "crrt=1 windows", "value": int(crrt_mask.sum())},
    {"check": "crrt=1 windows changed", "value": int((A[crrt_mask] != Bf[crrt_mask]).sum())},
    {"check": "epsilon", "value": EPS},
    {"check": "verdict", "value": verdict},
]).to_csv(OUT / "epsilon_floor_verification.csv", index=False)
print(f"\nVERDICT: {verdict}")
print(f"Wrote {OUT}/epsilon_floor_verification.csv")
