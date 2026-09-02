"""
Paper 2, Aim 4: produce one-step-ahead predicted probability distributions from
every model in the ladder, for MIMIC (train/test) and eICU.

The ladder is ordered by the information each predictor is allowed to use, so
that any gain can be attributed to a specific addition rather than to "a model":

  L0 persistence          next state = current state.  Not a model; the null.
  L1 transition_matrix    frozen P(s_t+1 | s_t), applied to the decoded state.
  L2 hmm_filtered         frozen P(s_t+1 | y_1..y_t) via forward filtering.
                          Same parameters as L1, but the full observation
                          history and a soft state posterior instead of a hard
                          decoded label.
  L3 multinom_logit       current state + the 21 model features at t.
  L4 grad_boosting        as L3, non-linear.
  L5 grad_boosting_plus   as L4 + age, sex, stroke subtype and window index.

L0-L2 involve no fitting at all: they are the discovery model used for a
different operation. L3-L5 are fitted on the MIMIC training split only and are
then frozen; eICU is touched once, at prediction time.

eICU carries a stroke subtype ("unspecified") that does not exist in MIMIC, so
subtype is one-hot encoded on the four MIMIC levels and an unspecified stay
takes an all-zero row. That is a documented limitation, not an imputation.

Output: prediction/pred_probs_{scope}.csv, one row per window pair, one column
per model x state.
"""
import copy
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
T1 = ROOT / "04_outputs/tables"
PRED = HERE / "prediction"
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401

EPS = 1e-4
SEED = 42
FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
LBL = pd.read_csv(T1 / "state_label_mapping.csv").set_index("raw_state")["name"].to_dict()
STATES = [LBL[i] for i in range(4)]                       # raw model order

# ---- the frozen model, floored exactly as in step 19 -------------------------
model = torch.load(ROOT / ".model_k4.pt", weights_only=False)
floored = copy.deepcopy(model)
for d in floored.distributions:
    for sub in d.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), EPS, 1 - EPS)
        p = p / p.sum(dim=-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False)
        sub._log_probs = torch.log(p)

TRANS = torch.exp(floored.edges).detach().numpy()          # raw state order
TRANS = TRANS / TRANS.sum(axis=1, keepdims=True)           # condition on not ending
print("frozen transition matrix, renormalised to exclude the end state:")
print(pd.DataFrame(np.round(TRANS, 4), index=STATES, columns=STATES).to_string())


def filtered_next(df, pid):
    """P(s_t+1 | y_1..y_t) for every window, by forward filtering.

    Sequences are batched by length: pomegranate needs equal-length inputs and
    the observation window count is small (2..12), so this is a handful of
    batches rather than one pass per patient.
    """
    df = df.sort_values([pid, "window_idx"])
    out = np.full((len(df), 4), np.nan)
    pos = {k: i for i, k in enumerate(df.index)}
    groups = df.groupby(pid, sort=False)
    by_len = {}
    for uid, g in groups:
        by_len.setdefault(len(g), []).append((uid, g))
    for L, items in sorted(by_len.items()):
        X = np.stack([g[FEATS].to_numpy(dtype=np.float32) for _, g in items])
        with torch.no_grad():
            alpha = floored.forward(torch.tensor(X)).numpy()      # (n, L, 4) log
        alpha = alpha - alpha.max(axis=2, keepdims=True)
        f = np.exp(alpha)
        f = f / f.sum(axis=2, keepdims=True)                      # P(s_t | y_1..t)
        nxt = f @ TRANS                                           # P(s_t+1 | y_1..t)
        nxt = nxt / nxt.sum(axis=2, keepdims=True)
        for k, (_, g) in enumerate(items):
            for j, idx in enumerate(g.index):
                out[pos[idx]] = nxt[k, j]
    return pd.DataFrame(out, index=df.index, columns=STATES)


def design(df, plus):
    """Feature matrix for the fitted models."""
    cur = pd.get_dummies(pd.Categorical(df.state, categories=STATES),
                         prefix="cur").astype(float)
    cur.index = df.index
    X = pd.concat([cur, df[FEATS].astype(float)], axis=1)
    if plus:
        sub = pd.get_dummies(pd.Categorical(df.stroke_subtype, categories=SUBTYPES),
                             prefix="sub").astype(float)
        sub.index = df.index
        X = pd.concat([X, df[["age", "female", "window_idx"]].astype(float), sub], axis=1)
    return X


SUBTYPES = ["AIS", "ICH", "SAH", "ICH+SAH"]                # MIMIC levels only

# ---- load pairs and attach the covariates L5 needs ---------------------------
mm = pd.read_csv(PRED / "mimic_pairs.csv")
ee = pd.read_csv(PRED / "eicu_pairs.csv")

mc = pd.read_csv(T1 / "patient_level_cohort.csv")
mc = mc[["stay_id", "age_at_adm_capped", "gender", "stroke_subtype"]].rename(
    columns={"stay_id": "uid", "age_at_adm_capped": "age"})
mc["female"] = (mc.gender == "F").astype(float)
mm = mm.merge(mc[["uid", "age", "female", "stroke_subtype"]], on="uid", how="left")

ec = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
ec = ec[["patientunitstayid", "age_n", "gender", "stroke_subtype"]].rename(
    columns={"patientunitstayid": "uid", "age_n": "age"})
ec["female"] = np.where(ec.gender == "Female", 1.0,
                        np.where(ec.gender == "Male", 0.0, np.nan))
ee = ee.merge(ec[["uid", "age", "female", "stroke_subtype"]], on="uid", how="left")

for d, lab in [(mm, "MIMIC"), (ee, "eICU")]:
    print(f"{lab}: age missing {d.age.isna().mean()*100:.2f}%, "
          f"sex missing {d.female.isna().mean()*100:.2f}%, "
          f"subtype outside MIMIC levels {(~d.stroke_subtype.isin(SUBTYPES)).mean()*100:.2f}%")

# ---- forward filtering needs every window; prediction needs the pairs --------
print("\nforward filtering...")
mm_f = filtered_next(mm.set_index(pd.RangeIndex(len(mm))), "uid")
ee_f = filtered_next(ee.set_index(pd.RangeIndex(len(ee))), "uid")
mm = mm.join(mm_f.add_prefix("L2|"))
ee = ee.join(ee_f.add_prefix("L2|"))

# ---- fit L3-L5 on the MIMIC training split only ------------------------------
tr = mm[(mm.split == "train") & mm.successor_contiguous].copy()
print(f"\nfitting on {len(tr):,} MIMIC training pairs from {tr.uid.nunique():,} stays")
y_tr = pd.Categorical(tr.next_state, categories=STATES).codes

fitted = {}
Xtr = design(tr, plus=False)
fitted["L3"] = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                  C=1.0, random_state=SEED).fit(Xtr, y_tr)
fitted["L4"] = HistGradientBoostingClassifier(random_state=SEED,
                                              early_stopping=True,
                                              validation_fraction=0.15).fit(Xtr, y_tr)
Xtr5 = design(tr, plus=True)
fitted["L5"] = HistGradientBoostingClassifier(random_state=SEED,
                                              early_stopping=True,
                                              validation_fraction=0.15).fit(Xtr5, y_tr)
for k, m_ in fitted.items():
    print(f"  {k}: {type(m_).__name__}, {Xtr5.shape[1] if k=='L5' else Xtr.shape[1]} inputs")

# ---- emit predictions for every evaluation scope -----------------------------
def emit(df, scope):
    d = df[df.successor_contiguous].copy()
    cur = pd.get_dummies(pd.Categorical(d.state, categories=STATES)).astype(float).to_numpy()
    P = {"L0": cur,
         "L1": cur @ TRANS,
         "L2": d[[f"L2|{s}" for s in STATES]].to_numpy()}
    P["L1"] = P["L1"] / P["L1"].sum(axis=1, keepdims=True)
    P["L3"] = fitted["L3"].predict_proba(design(d, plus=False))
    P["L4"] = fitted["L4"].predict_proba(design(d, plus=False))
    P["L5"] = fitted["L5"].predict_proba(design(d, plus=True))
    out = d[["uid", "window_idx", "state", "next_state"]].copy()
    for k, M in P.items():
        assert M.shape == (len(d), 4), f"{scope} {k}: {M.shape}"
        for j, s in enumerate(STATES):
            out[f"{k}|{s}"] = M[:, j]
    out.to_csv(PRED / f"pred_probs_{scope}.csv", index=False)
    print(f"  {scope:<26} {len(out):>7,} pairs -> pred_probs_{scope}.csv")
    return out


print("\nwriting predictions:")
emit(mm[mm.split == "test"], "mimic-test")
emit(ee, "eicu-full")
emit(ee[ee.gcs_eligible], "eicu-gcs-eligible")
emit(ee[ee.vent_ascertainable], "eicu-vent-ascertainable")
emit(ee[ee.gcs_eligible & ee.vent_ascertainable], "eicu-both")
print("\ndone")
