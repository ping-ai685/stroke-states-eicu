"""
Paper 2, Aim 4: paired contrasts between adjacent rungs of the prediction ladder.

Non-overlapping confidence intervals are not a test of a difference, and the
models are evaluated on the same window pairs, so the difference is estimated
directly: the same patient-clustered resample is applied to both models and
delta AUROC is recomputed within it. The interval is therefore paired and much
tighter than the two marginal intervals would suggest.

Each contrast isolates one addition to the information set:

  L1 - L0   does the frozen transition structure beat "no change"?
  L2 - L1   does the observation history add over the decoded state label?
  L3 - L2   does fitting a predictor add over the frozen discovery model?
  L4 - L3   does relaxing linearity add?
  L5 - L4   do baseline covariates add?
  L5 - L2   total gain from fitting anything at all.

Output: prediction/prediction_contrasts.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
PRED = HERE / "prediction"
T1 = HERE.parent / "04_outputs/tables"
STATES = pd.read_csv(T1 / "state_label_mapping.csv").set_index("raw_state")["name"].reindex(range(4)).tolist()
SCOPES = ["mimic-test", "eicu-full", "eicu-gcs-eligible",
          "eicu-vent-ascertainable", "eicu-both"]
CONTRASTS = [("L1", "L0"), ("L2", "L1"), ("L3", "L2"),
             ("L4", "L3"), ("L5", "L4"), ("L5", "L2")]
NBOOT = 1000
RNG = np.random.default_rng(7)

rows = []
for scope in SCOPES:
    d = pd.read_csv(PRED / f"pred_probs_{scope}.csv")
    cur = pd.Categorical(d.state, categories=STATES).codes
    changed = (d.state != d.next_state).to_numpy().astype(int)
    uids = d.uid.to_numpy()
    uniq, inv = np.unique(uids, return_inverse=True)
    blocks = [np.flatnonzero(inv == i) for i in range(len(uniq))]

    def chg_score(key):
        P = d[[f"{key}|{s}" for s in STATES]].to_numpy()
        return np.round(1.0 - P[np.arange(len(P)), cur], 12)

    S = {k: chg_score(k) for k in ["L0", "L1", "L2", "L3", "L4", "L5"]}

    # one set of resamples, reused for every contrast in this scope
    picks = [np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))])
             for _ in range(NBOOT)]

    for a, b in CONTRASTS:
        obs = roc_auc_score(changed, S[a]) - roc_auc_score(changed, S[b])
        deltas = []
        for idx in picks:
            y = changed[idx]
            if y.all() or not y.any():
                continue
            deltas.append(roc_auc_score(y, S[a][idx]) - roc_auc_score(y, S[b][idx]))
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append({"scope": scope, "contrast": f"{a} - {b}",
                     "delta_auroc": round(float(obs), 4),
                     "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                     "excludes_zero": bool(lo > 0 or hi < 0),
                     "p_boot_two_sided": round(float(2 * min((deltas <= 0).mean(),
                                                             (deltas >= 0).mean())), 4)})
    print(f"  {scope} done")

res = pd.DataFrame(rows)
res.to_csv(PRED / "prediction_contrasts.csv", index=False)
pd.set_option("display.width", 160)
for scope in SCOPES:
    print(f"\n=== {scope} ===")
    print(res[res.scope == scope][["contrast", "delta_auroc", "lo", "hi",
                                   "excludes_zero", "p_boot_two_sided"]].to_string(index=False))
print(f"\nwrote {PRED/'prediction_contrasts.csv'}")
