"""
Paper 2, Aim 4: evaluate the prediction ladder.

Overall accuracy is reported but is not the headline, because 90.7% of window
pairs do not change state: a rule that predicts "same as now" and does no
modelling at all scores about 0.91. The question with clinical content is
whether a model can tell which patients are about to change, so the primary
metric is discrimination for the binary event "state differs in the next
window", where the null predictor is by construction uninformative.

Confidence intervals are patient-clustered bootstrap percentiles: windows from
one stay are resampled together, because they are not independent.

Output: prediction/metrics_by_model.csv, prediction/metrics_change_detection.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             balanced_accuracy_score)

HERE = Path(__file__).parent
PRED = HERE / "prediction"
T1 = HERE.parent / "04_outputs/tables"
STATES = pd.read_csv(T1 / "state_label_mapping.csv").set_index("raw_state")["name"].reindex(range(4)).tolist()
MODELS = {"L0": "persistence (null)", "L1": "frozen transition matrix",
          "L2": "frozen HMM, forward filtered", "L3": "multinomial logistic",
          "L4": "gradient boosting", "L5": "gradient boosting + covariates"}
SCOPES = ["mimic-test", "eicu-full", "eicu-gcs-eligible",
          "eicu-vent-ascertainable", "eicu-both"]
NBOOT, EPS, RNG = 300, 1e-6, np.random.default_rng(42)
# Only the primary metrics are bootstrapped: the multiclass OvR AUC costs
# more than the rest combined and is reported as a point estimate.
BOOT_KEYS = ["accuracy", "change_auroc", "change_auprc", "change_sens_at_90spec"]


def clip(P):
    """Only for log loss. Renormalising after clipping perturbs equal
    probabilities in the last bits, and a rank-based metric will happily read
    that noise as signal -- the persistence null scored AUROC 0.57 instead of
    0.50 until this was separated out."""
    P = np.clip(P, EPS, 1 - EPS)
    return P / P.sum(axis=1, keepdims=True)


def sens_at_spec(y, s, spec=0.90):
    neg = np.sort(s[y == 0])
    if not len(neg):
        return np.nan
    thr = neg[int(np.ceil(spec * len(neg))) - 1]
    return float((s[y == 1] > thr).mean())


def point_metrics(y_idx, P, Pc, changed, score, full=True):
    """P is the raw probability matrix (ranks, argmax); Pc is the clipped one
    (log loss only)."""
    pred = P.argmax(axis=1)
    m = {
        "accuracy": float((pred == y_idx).mean()),
        "change_auroc": float(roc_auc_score(changed, score)) if changed.any() and not changed.all() else np.nan,
        "change_auprc": float(average_precision_score(changed, score)),
        "change_sens_at_90spec": sens_at_spec(changed, score),
    }
    if full:
        m["balanced_accuracy"] = float(balanced_accuracy_score(y_idx, pred))
        m["macro_auc_ovr"] = float(roc_auc_score(y_idx, P, multi_class="ovr",
                                                 average="macro", labels=list(range(4))))
        m["log_loss"] = float(log_loss(y_idx, Pc, labels=list(range(4))))
    return m


rows = []
for scope in SCOPES:
    d = pd.read_csv(PRED / f"pred_probs_{scope}.csv")
    y_idx = pd.Categorical(d.next_state, categories=STATES).codes
    cur_idx = pd.Categorical(d.state, categories=STATES).codes
    changed = (d.state != d.next_state).to_numpy().astype(int)
    uids = d.uid.to_numpy()
    uniq = np.unique(uids)
    # index of each stay's rows, for the clustered bootstrap
    order = np.argsort(uids, kind="stable")
    starts = np.searchsorted(uids[order], uniq)
    ends = np.append(starts[1:], len(uids))
    blocks = [order[a:b] for a, b in zip(starts, ends)]

    for key, name in MODELS.items():
        P = d[[f"{key}|{s}" for s in STATES]].to_numpy()
        Pc = clip(P)
        # P(the state changes) = 1 - P(next == current). Rounded because ties
        # must stay ties: see clip().
        score = np.round(1.0 - P[np.arange(len(P)), cur_idx], 12)
        pt = point_metrics(y_idx, P, Pc, changed, score)
        if key == "L0":
            # The null assigns the same P(change) to every window, so its AUROC is
            # 0.5 by construction. Anything else means the current-state column and
            # the state labels have come apart -- the failure mode that scrambled
            # the first run of this analysis.
            assert abs(pt["change_auroc"] - 0.5) < 1e-6, (
                f"{scope}: persistence AUROC is {pt['change_auroc']:.4f}, not 0.5 "
                "-- state labels and probability columns are misaligned")

        boot = {k: [] for k in BOOT_KEYS}
        for _ in range(NBOOT):
            pick = RNG.integers(0, len(blocks), len(blocks))
            idx = np.concatenate([blocks[i] for i in pick])
            try:
                b = point_metrics(y_idx[idx], P[idx], Pc[idx], changed[idx], score[idx], full=False)
            except ValueError:
                continue
            for k in BOOT_KEYS:
                boot[k].append(b[k])
        rec = {"scope": scope, "model": key, "model_name": name,
               "pairs": len(d), "patients": len(uniq),
               "changed_pct": round(changed.mean() * 100, 2)}
        for k, v in pt.items():
            rec[k] = round(v, 4)
            if k in boot:
                arr = np.array([x for x in boot[k] if np.isfinite(x)])
                lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
                rec[f"{k}_lo"] = round(float(lo), 4)
                rec[f"{k}_hi"] = round(float(hi), 4)
        rows.append(rec)
    print(f"  {scope:<26} {len(d):>7,} pairs, {len(uniq):>6,} patients  done")

res = pd.DataFrame(rows)
res.to_csv(PRED / "metrics_by_model.csv", index=False)

pd.set_option("display.width", 200)
for scope in SCOPES:
    s = res[res.scope == scope]
    print(f"\n=== {scope}  ({int(s.pairs.iloc[0]):,} pairs, "
          f"{s.changed_pct.iloc[0]}% changed) ===")
    show = s[["model_name", "accuracy", "change_auroc", "change_auroc_lo",
              "change_auroc_hi", "change_auprc", "change_sens_at_90spec",
              "log_loss"]].copy()
    show.columns = ["model", "acc", "chg AUROC", "lo", "hi", "chg AUPRC",
                    "sens@90spec", "logloss"]
    print(show.to_string(index=False))
print(f"\nwrote {PRED/'metrics_by_model.csv'}")
