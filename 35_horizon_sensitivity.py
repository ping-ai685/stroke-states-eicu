"""
Paper 2, Aim 4: separate the horizon effect from the selection effect.

Discrimination rises with the horizon, which is the opposite of the usual
pattern and must not be reported as "the model predicts better further out".
Two things move together: the event becomes more common, and the eligible set
shrinks to stays that lasted long enough, which selects for longer stays.

This script holds the population fixed. Every horizon is re-evaluated on the
window pairs that are eligible at the LONGEST horizon, so the same patients and
the same index windows are used throughout and only the target moves.

Also reports, per horizon, the share of the discrimination above the null that
the frozen model reaches without any fitting.

Output: prediction/horizon_sensitivity.csv
"""
import copy, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent; ROOT = HERE.parent
T1 = ROOT / "04_outputs/tables"; PRED = HERE / "prediction"
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401

EPS, SEED, NBOOT = 1e-4, 42, 300
HORIZONS = [1, 2, 4]
FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
STATES = pd.read_csv(T1/"state_label_mapping.csv").set_index("raw_state")["name"].reindex(range(4)).tolist()
SUBTYPES = ["AIS", "ICH", "SAH", "ICH+SAH"]
RNG = np.random.default_rng(SEED)

model = torch.load(ROOT / ".model_k4.pt", weights_only=False)
fl = copy.deepcopy(model)
for d_ in fl.distributions:
    for sub in d_.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), EPS, 1-EPS); p = p/p.sum(-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False); sub._log_probs = torch.log(p)
TRANS = torch.exp(fl.edges).detach().numpy(); TRANS = TRANS/TRANS.sum(1, keepdims=True)

M = pd.read_csv(PRED/"multihorizon_metrics.csv")
print("=== share of discrimination above the null reached with no fitting (L2 vs best) ===")
share_rows = []
for scope in M.scope.unique():
    for h in sorted(M.horizon_h.unique()):
        s = M[(M.scope==scope)&(M.horizon_h==h)].set_index("model")
        best = s.loc[["L3","L4","L5"], "change_auroc"].max()
        g2, gb = s.loc["L2","change_auroc"]-0.5, best-0.5
        share_rows.append({"scope":scope, "horizon_h":h, "L2":s.loc["L2","change_auroc"],
                           "best_fitted":round(float(best),4),
                           "gap":round(float(best-s.loc["L2","change_auroc"]),4),
                           "share_pct":round(float(g2/gb*100),1)})
S = pd.DataFrame(share_rows)
print(S.pivot(index="scope", columns="horizon_h", values="share_pct").to_string())
print("\ngap between the frozen model and the best fitted model:")
print(S.pivot(index="scope", columns="horizon_h", values="gap").to_string())

# ---- population held fixed at the 24 h eligible set --------------------------
mm = pd.read_csv(PRED/"mimic_pairs.csv"); ee = pd.read_csv(PRED/"eicu_pairs.csv")
mc = pd.read_csv(T1/"patient_level_cohort.csv")[["stay_id","age_at_adm_capped","gender","stroke_subtype"]]
mc = mc.rename(columns={"stay_id":"uid","age_at_adm_capped":"age"}); mc["female"]=(mc.gender=="F").astype(float)
mm = mm.merge(mc[["uid","age","female","stroke_subtype"]], on="uid", how="left")
ec = pd.read_csv(HERE/"cohort/patient_level_cohort.csv")[["patientunitstayid","age_n","gender","stroke_subtype"]]
ec = ec.rename(columns={"patientunitstayid":"uid","age_n":"age"})
ec["female"]=np.where(ec.gender=="Female",1.0,np.where(ec.gender=="Male",0.0,np.nan))
ee = ee.merge(ec[["uid","age","female","stroke_subtype"]], on="uid", how="left")

def filt(df, pid):
    df = df.sort_values([pid,"window_idx"]); out=np.full((len(df),4),np.nan)
    pos={k:i for i,k in enumerate(df.index)}; by={}
    for uid,g in df.groupby(pid, sort=False): by.setdefault(len(g),[]).append((uid,g))
    for L,items in sorted(by.items()):
        X=np.stack([g[FEATS].to_numpy(dtype=np.float32) for _,g in items])
        with torch.no_grad(): a=fl.forward(torch.tensor(X)).numpy()
        a=a-a.max(2,keepdims=True); f=np.exp(a); f=f/f.sum(2,keepdims=True)
        for k,(_,g) in enumerate(items):
            for j,idx in enumerate(g.index): out[pos[idx]]=f[k,j]
    return pd.DataFrame(out, index=df.index, columns=[f"f|{s}" for s in STATES])

print("\nforward filtering...")
mm = mm.reset_index(drop=True).join(filt(mm.reset_index(drop=True),"uid"))
ee = ee.reset_index(drop=True).join(filt(ee.reset_index(drop=True),"uid"))
FC=[f"f|{s}" for s in STATES]

def design(d, plus):
    cur=pd.get_dummies(pd.Categorical(d.state,categories=STATES),prefix="cur").astype(float); cur.index=d.index
    X=pd.concat([cur, d[FEATS].astype(float)],axis=1)
    if plus:
        sub=pd.get_dummies(pd.Categorical(d.stroke_subtype,categories=SUBTYPES),prefix="sub").astype(float); sub.index=d.index
        X=pd.concat([X, d[["age","female","window_idx"]].astype(float), sub],axis=1)
    return X

def targets(df, h):
    d=df.sort_values(["uid","window_idx"]).copy()
    d["target"]=d.groupby("uid").state.shift(-h); d["twi"]=d.groupby("uid").window_idx.shift(-h)
    return d[d.twi==d.window_idx+h]

HMAX=max(HORIZONS)
keep_m=set(map(tuple, targets(mm,HMAX)[["uid","window_idx"]].to_numpy()))
keep_e=set(map(tuple, targets(ee,HMAX)[["uid","window_idx"]].to_numpy()))
print(f"\npopulation held fixed at the {HMAX*6} h eligible set: "
      f"MIMIC {len(keep_m):,} index windows, eICU {len(keep_e):,}")

rows=[]
for h in HORIZONS:
    Mh, Eh = targets(mm,h), targets(ee,h)
    Mh = Mh[[tuple(x) in keep_m for x in Mh[["uid","window_idx"]].to_numpy()]]
    Eh = Eh[[tuple(x) in keep_e for x in Eh[["uid","window_idx"]].to_numpy()]]
    Th=np.linalg.matrix_power(TRANS,h)
    tr=Mh[Mh.split=="train"]; y=pd.Categorical(tr.target,categories=STATES).codes
    f5=HistGradientBoostingClassifier(random_state=SEED,early_stopping=True,
                                      validation_fraction=0.15).fit(design(tr,True),y)
    for scope,d in [("mimic-test",Mh[Mh.split=="test"]),("eicu-full",Eh)]:
        cur=pd.Categorical(d.state,categories=STATES).codes
        chg=(d.state!=d.target).to_numpy().astype(int)
        P={"L0":pd.get_dummies(pd.Categorical(d.state,categories=STATES)).astype(float).to_numpy()}
        P["L1"]=P["L0"]@Th; P["L2"]=d[FC].to_numpy()@Th; P["L5"]=f5.predict_proba(design(d,True))
        r={"horizon_h":h*6,"scope":scope,"pairs":len(d),"changed_pct":round(chg.mean()*100,2)}
        for k,Q in P.items():
            Q=Q/Q.sum(1,keepdims=True); sc=np.round(1.0-Q[np.arange(len(Q)),cur],12)
            r[k]=round(float(roc_auc_score(chg,sc)),4)
        rows.append(r)
        print(f"  h={h*6:>2}h {scope:<11} {len(d):>7,} pairs  {chg.mean()*100:>5.1f}% changed  "
              f"L2 {r['L2']:.4f}  L5 {r['L5']:.4f}")

F=pd.DataFrame(rows)
out=pd.concat([S.assign(analysis="all eligible pairs"),
               F.assign(analysis="population fixed at 24 h")], ignore_index=True)
out.to_csv(PRED/"horizon_sensitivity.csv", index=False)
print("\n=== horizon effect with the population held fixed ===")
for scope in F.scope.unique():
    print(f"\n{scope}")
    print(F[F.scope==scope][["horizon_h","pairs","changed_pct","L0","L1","L2","L5"]].to_string(index=False))
print(f"\nwrote {PRED/'horizon_sensitivity.csv'}")
