"""
Paper 2, Aim 4: how far ahead does the state representation carry information?

The one-step analysis asks about the next 6-hour window. Six hours is a short
lead time, so the same ladder is run at 12 h (two windows) and 24 h (four
windows). Nothing is refitted for L0-L2: the frozen transition matrix is raised
to the h-th power, which is exactly the model's own statement about h steps
ahead. L3-L5 are refitted per horizon on the MIMIC training split, because their
target changes.

Two things move as the horizon lengthens and both are reported, because either
one alone would mislead:
  - more patients have left the ICU, so the eligible set shrinks and selects
    for longer stays;
  - more of the remaining patients have changed state, so the event being
    predicted is less rare and the persistence null gets weaker.

Output: prediction/multihorizon_metrics.csv, prediction/multihorizon_qa.csv
"""
import copy
import sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

HERE = Path(__file__).parent
ROOT = HERE.parent
T1 = ROOT / "04_outputs/tables"
PRED = HERE / "prediction"
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401


def sens_at_90spec(y, score):
    """Sensitivity at the threshold giving 90% specificity.

    AUROC summarises every threshold at once; a clinician uses one. Reporting
    only AUROC across horizons leaves open whether a longer horizon is actually
    more usable or merely easier to rank.
    """
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.interp(0.10, fpr, tpr))

EPS, SEED, NBOOT = 1e-4, 42, 200
HORIZONS = [1, 2, 4]                                       # windows: 6 h, 12 h, 24 h
SCOPES = ["mimic-test", "eicu-full", "eicu-both"]
FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
_LM = pd.read_csv(T1 / "state_label_mapping.csv")
STATES = _LM.set_index("raw_state")["name"].reindex(range(4)).tolist()
SUBTYPES = ["AIS", "ICH", "SAH", "ICH+SAH"]
RNG = np.random.default_rng(SEED)

model = torch.load(ROOT / ".model_k4.pt", weights_only=False)
floored = copy.deepcopy(model)
for d_ in floored.distributions:
    for sub in d_.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), EPS, 1 - EPS)
        p = p / p.sum(dim=-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False)
        sub._log_probs = torch.log(p)
TRANS = torch.exp(floored.edges).detach().numpy()
TRANS = TRANS / TRANS.sum(axis=1, keepdims=True)


def filtered_posterior(df, pid):
    """P(s_t | y_1..y_t), cached because it does not depend on the horizon."""
    df = df.sort_values([pid, "window_idx"])
    out = np.full((len(df), 4), np.nan)
    pos = {k: i for i, k in enumerate(df.index)}
    by_len = {}
    for uid, g in df.groupby(pid, sort=False):
        by_len.setdefault(len(g), []).append((uid, g))
    for L, items in sorted(by_len.items()):
        X = np.stack([g[FEATS].to_numpy(dtype=np.float32) for _, g in items])
        with torch.no_grad():
            a = floored.forward(torch.tensor(X)).numpy()
        a = a - a.max(axis=2, keepdims=True)
        f = np.exp(a)
        f = f / f.sum(axis=2, keepdims=True)
        for k, (_, g) in enumerate(items):
            for j, idx in enumerate(g.index):
                out[pos[idx]] = f[k, j]
    return pd.DataFrame(out, index=df.index, columns=[f"f|{s}" for s in STATES])


def design(d, plus):
    cur = pd.get_dummies(pd.Categorical(d.state, categories=STATES), prefix="cur").astype(float)
    cur.index = d.index
    X = pd.concat([cur, d[FEATS].astype(float)], axis=1)
    if plus:
        sub = pd.get_dummies(pd.Categorical(d.stroke_subtype, categories=SUBTYPES), prefix="sub").astype(float)
        sub.index = d.index
        X = pd.concat([X, d[["age", "female", "window_idx"]].astype(float), sub], axis=1)
    return X


# ---- data --------------------------------------------------------------------
mm = pd.read_csv(PRED / "mimic_pairs.csv")
ee = pd.read_csv(PRED / "eicu_pairs.csv")
mc = pd.read_csv(T1 / "patient_level_cohort.csv")[["stay_id", "age_at_adm_capped", "gender", "stroke_subtype"]]
mc = mc.rename(columns={"stay_id": "uid", "age_at_adm_capped": "age"})
mc["female"] = (mc.gender == "F").astype(float)
mm = mm.merge(mc[["uid", "age", "female", "stroke_subtype"]], on="uid", how="left")
ec = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")[["patientunitstayid", "age_n", "gender", "stroke_subtype"]]
ec = ec.rename(columns={"patientunitstayid": "uid", "age_n": "age"})
ec["female"] = np.where(ec.gender == "Female", 1.0, np.where(ec.gender == "Male", 0.0, np.nan))
ee = ee.merge(ec[["uid", "age", "female", "stroke_subtype"]], on="uid", how="left")

print("forward filtering (once, reused for every horizon)...")
mm = mm.reset_index(drop=True).join(filtered_posterior(mm.reset_index(drop=True), "uid"))
ee = ee.reset_index(drop=True).join(filtered_posterior(ee.reset_index(drop=True), "uid"))
FCOLS = [f"f|{s}" for s in STATES]


def add_target(df, h):
    d = df.sort_values(["uid", "window_idx"]).copy()
    d["target"] = d.groupby("uid").state.shift(-h)
    d["target_widx"] = d.groupby("uid").window_idx.shift(-h)
    d = d[d.target_widx == d.window_idx + h].copy()      # the horizon must be intact
    d["changed"] = (d.state != d.target).astype(int)
    return d


qa, rows = [], []
for h in HORIZONS:
    hours = h * 6
    M, E = add_target(mm, h), add_target(ee, h)
    Th = np.linalg.matrix_power(TRANS, h)
    tr = M[M.split == "train"]
    y_tr = pd.Categorical(tr.target, categories=STATES).codes
    fit = {"L3": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED).fit(design(tr, False), y_tr),
           "L4": HistGradientBoostingClassifier(random_state=SEED, early_stopping=True,
                                                validation_fraction=0.15).fit(design(tr, False), y_tr),
           "L5": HistGradientBoostingClassifier(random_state=SEED, early_stopping=True,
                                                validation_fraction=0.15).fit(design(tr, True), y_tr)}
    print(f"\nhorizon {hours:>2} h: {len(tr):,} training pairs")

    for scope in SCOPES:
        d = {"mimic-test": M[M.split == "test"],
             "eicu-full": E,
             "eicu-both": E[E.gcs_eligible & E.vent_ascertainable]}[scope]
        cur = pd.Categorical(d.state, categories=STATES).codes
        y = pd.Categorical(d.target, categories=STATES).codes
        chg = d.changed.to_numpy()
        onehot = pd.get_dummies(pd.Categorical(d.state, categories=STATES)).astype(float).to_numpy()
        P = {"L0": onehot, "L1": onehot @ Th, "L2": d[FCOLS].to_numpy() @ Th,
             "L3": fit["L3"].predict_proba(design(d, False)),
             "L4": fit["L4"].predict_proba(design(d, False)),
             "L5": fit["L5"].predict_proba(design(d, True))}
        qa.append({"horizon_h": hours, "scope": scope, "pairs": len(d),
                   "patients": d.uid.nunique(), "changed_pct": round(chg.mean() * 100, 2),
                   "pairs_vs_6h": None})
        uids = d.uid.to_numpy()
        _, inv = np.unique(uids, return_inverse=True)
        blocks = [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]
        picks = [np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))])
                 for _ in range(NBOOT)]
        for k, Q in P.items():
            Q = Q / Q.sum(axis=1, keepdims=True)
            sc = np.round(1.0 - Q[np.arange(len(Q)), cur], 12)
            auc = roc_auc_score(chg, sc) if 0 < chg.mean() < 1 else np.nan
            if k == "L0":
                assert abs(auc - 0.5) < 1e-6, f"{scope} h={h}: null AUROC {auc}"
            b = [roc_auc_score(chg[i], sc[i]) for i in picks if 0 < chg[i].mean() < 1]
            lo, hi = np.percentile(b, [2.5, 97.5])
            rows.append({"horizon_h": hours, "scope": scope, "model": k,
                         "pairs": len(d), "changed_pct": round(chg.mean() * 100, 2),
                         "accuracy": round(float((Q.argmax(1) == y).mean()), 4),
                         "change_auroc": round(float(auc), 4),
                         "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                         "change_auprc": round(float(average_precision_score(chg, sc)), 4),
                         "change_sens_at_90spec": round(float(sens_at_90spec(chg, sc)), 4)})
        print(f"  {scope:<12} {len(d):>7,} pairs  {chg.mean()*100:>5.1f}% changed")

R = pd.DataFrame(rows)
R.to_csv(PRED / "multihorizon_metrics.csv", index=False)
pd.DataFrame(qa).to_csv(PRED / "multihorizon_qa.csv", index=False)

pd.set_option("display.width", 200)
NAME = {"L0": "persistence (null)", "L1": "frozen transition matrix",
        "L2": "frozen HMM, forward filtered", "L3": "multinomial logistic",
        "L4": "gradient boosting", "L5": "gradient boosting + covariates"}
for scope in SCOPES:
    print(f"\n=== {scope}: change-detection AUROC by horizon ===")
    t = R[R.scope == scope].pivot(index="model", columns="horizon_h", values="change_auroc")
    n = R[R.scope == scope].groupby("horizon_h").agg(pairs=("pairs", "first"),
                                                     changed=("changed_pct", "first"))
    t.index = [NAME[i] for i in t.index]
    print(t.to_string())
    print(n.to_string())
print(f"\nwrote {PRED/'multihorizon_metrics.csv'}")
