"""
Paper 2, Aim 4: the full metric set for the sequence model (L6).

Step 38 persists the seed-averaged predicted distributions, so this computes the
same quantities reported for every other rung of the ladder without retraining.
Kept separate from step 31 because L6 is evaluated per horizon and its
predictions live in their own files.

Output: prediction/metrics_L6.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

HERE = Path(__file__).parent
PRED = HERE / "prediction"
T1 = HERE.parent / "04_outputs/tables"
STATES = pd.read_csv(T1/"state_label_mapping.csv").set_index("raw_state")["name"].reindex(range(4)).tolist()
NBOOT, EPS = 300, 1e-6
RNG = np.random.default_rng(42)


def sens_at_spec(y, s, spec=0.90):
    neg = np.sort(s[y == 0])
    if not len(neg):
        return np.nan
    return float((s[y == 1] > neg[int(np.ceil(spec * len(neg))) - 1]).mean())


rows = []
for scope in ["mimic-test", "eicu-full"]:
    for h in [6, 12, 24]:
        d = pd.read_csv(PRED / f"pred_probs_L6_{scope}_h{h}.csv")
        P = d[[f"L6|{s}" for s in STATES]].to_numpy()
        cur = pd.Categorical(d.state, categories=STATES).codes
        y = pd.Categorical(d.target, categories=STATES).codes
        chg = (d.state != d.target).to_numpy().astype(int)
        score = np.round(1.0 - P[np.arange(len(P)), cur], 12)
        Pc = np.clip(P, EPS, 1 - EPS); Pc = Pc / Pc.sum(1, keepdims=True)

        uids = d.uid.to_numpy()
        _, inv = np.unique(uids, return_inverse=True)
        blocks = [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]
        b = []
        for _ in range(NBOOT):
            ix = np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))])
            if 0 < chg[ix].mean() < 1:
                b.append(roc_auc_score(chg[ix], score[ix]))
        lo, hi = np.percentile(b, [2.5, 97.5])
        rows.append({
            "scope": scope, "horizon_h": h, "model": "L6",
            "model_name": "sequence model (GRU)", "pairs": len(d),
            "changed_pct": round(float(chg.mean() * 100), 2),
            "accuracy": round(float((P.argmax(1) == y).mean()), 4),
            "change_auroc": round(float(roc_auc_score(chg, score)), 4),
            "change_auroc_lo": round(float(lo), 4),
            "change_auroc_hi": round(float(hi), 4),
            "change_auprc": round(float(average_precision_score(chg, score)), 4),
            "change_sens_at_90spec": round(float(sens_at_spec(chg, score)), 4),
            "log_loss": round(float(log_loss(y, Pc, labels=list(range(4)))), 4),
        })
        print(f"  {scope:<11} h={h:>2}  n={len(d):>6,}  AUROC {rows[-1]['change_auroc']:.3f} "
              f"({lo:.3f}–{hi:.3f})  AUPRC {rows[-1]['change_auprc']:.3f}  "
              f"sens@90 {rows[-1]['change_sens_at_90spec']*100:.1f}%  acc {rows[-1]['accuracy']:.3f}")

M = pd.DataFrame(rows)
M.to_csv(PRED / "metrics_L6.csv", index=False)
print(f"\nwrote {PRED/'metrics_L6.csv'}")
