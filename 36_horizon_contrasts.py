"""
Paper 2, Aim 4: paired contrasts at each horizon.

Step 32 tested the ladder at 6 h only. The claim that fitting adds almost
nothing at 24 h rests on a difference of 0.005 AUROC, which cannot be reported
without an interval: a difference that small is only interesting if it is
bounded, and it matters whether the bound excludes zero.

Contrasts, at 6 h, 12 h and 24 h:
  L2 - L0   what the frozen representation carries over the null
  L5 - L2   what fitting adds over the frozen representation

Output: prediction/horizon_contrasts.csv
"""
import copy, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent; ROOT = HERE.parent
T1 = ROOT / "04_outputs/tables"; PRED = HERE / "prediction"
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401

EPS, SEED, NBOOT = 1e-4, 42, 500
HORIZONS = [1, 2, 4]
FEATS = pd.read_csv(HERE/"frozen_params/feature_order.csv").feature.tolist()
STATES = pd.read_csv(T1/"state_label_mapping.csv").set_index("raw_state")["name"].reindex(range(4)).tolist()
SUBTYPES = ["AIS","ICH","SAH","ICH+SAH"]
RNG = np.random.default_rng(SEED)

model = torch.load(ROOT/".model_k4.pt", weights_only=False)
fl = copy.deepcopy(model)
for d_ in fl.distributions:
    for sub in d_.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), EPS, 1-EPS); p = p/p.sum(-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False); sub._log_probs = torch.log(p)
TRANS = torch.exp(fl.edges).detach().numpy(); TRANS = TRANS/TRANS.sum(1, keepdims=True)

mm = pd.read_csv(PRED/"mimic_pairs.csv"); ee = pd.read_csv(PRED/"eicu_pairs.csv")
mc = pd.read_csv(T1/"patient_level_cohort.csv")[["stay_id","age_at_adm_capped","gender","stroke_subtype"]]
mc = mc.rename(columns={"stay_id":"uid","age_at_adm_capped":"age"}); mc["female"]=(mc.gender=="F").astype(float)
mm = mm.merge(mc[["uid","age","female","stroke_subtype"]], on="uid", how="left")
ec = pd.read_csv(HERE/"cohort/patient_level_cohort.csv")[["patientunitstayid","age_n","gender","stroke_subtype"]]
ec = ec.rename(columns={"patientunitstayid":"uid","age_n":"age"})
ec["female"]=np.where(ec.gender=="Female",1.0,np.where(ec.gender=="Male",0.0,np.nan))
ee = ee.merge(ec[["uid","age","female","stroke_subtype"]], on="uid", how="left")

def filt(df, pid):
    df=df.sort_values([pid,"window_idx"]); out=np.full((len(df),4),np.nan)
    pos={k:i for i,k in enumerate(df.index)}; by={}
    for uid,g in df.groupby(pid,sort=False): by.setdefault(len(g),[]).append((uid,g))
    for L,items in sorted(by.items()):
        X=np.stack([g[FEATS].to_numpy(dtype=np.float32) for _,g in items])
        with torch.no_grad(): a=fl.forward(torch.tensor(X)).numpy()
        a=a-a.max(2,keepdims=True); f=np.exp(a); f=f/f.sum(2,keepdims=True)
        for k,(_,g) in enumerate(items):
            for j,idx in enumerate(g.index): out[pos[idx]]=f[k,j]
    return pd.DataFrame(out, index=df.index, columns=[f"f|{s}" for s in STATES])

print("forward filtering...")
mm = mm.reset_index(drop=True).join(filt(mm.reset_index(drop=True),"uid"))
ee = ee.reset_index(drop=True).join(filt(ee.reset_index(drop=True),"uid"))
FC=[f"f|{s}" for s in STATES]

def design(d, plus):
    cur=pd.get_dummies(pd.Categorical(d.state,categories=STATES),prefix="cur").astype(float); cur.index=d.index
    X=pd.concat([cur,d[FEATS].astype(float)],axis=1)
    if plus:
        sub=pd.get_dummies(pd.Categorical(d.stroke_subtype,categories=SUBTYPES),prefix="sub").astype(float); sub.index=d.index
        X=pd.concat([X,d[["age","female","window_idx"]].astype(float),sub],axis=1)
    return X

def targets(df,h):
    d=df.sort_values(["uid","window_idx"]).copy()
    d["target"]=d.groupby("uid").state.shift(-h); d["twi"]=d.groupby("uid").window_idx.shift(-h)
    return d[d.twi==d.window_idx+h]

rows=[]
for h in HORIZONS:
    M,E = targets(mm,h), targets(ee,h)
    Th = np.linalg.matrix_power(TRANS,h)
    tr = M[M.split=="train"]; y=pd.Categorical(tr.target,categories=STATES).codes
    f5 = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True,
                                        validation_fraction=0.15).fit(design(tr,True), y)
    for scope,d in [("mimic-test",M[M.split=="test"]),("eicu-full",E)]:
        cur=pd.Categorical(d.state,categories=STATES).codes
        chg=(d.state!=d.target).to_numpy().astype(int)
        oh=pd.get_dummies(pd.Categorical(d.state,categories=STATES)).astype(float).to_numpy()
        S={}
        for k,Q in [("L0",oh),("L2",d[FC].to_numpy()@Th),("L5",f5.predict_proba(design(d,True)))]:
            Q=Q/Q.sum(1,keepdims=True); S[k]=np.round(1.0-Q[np.arange(len(Q)),cur],12)
        uids=d.uid.to_numpy(); _,inv=np.unique(uids,return_inverse=True)
        blocks=[np.flatnonzero(inv==i) for i in range(inv.max()+1)]
        picks=[np.concatenate([blocks[i] for i in RNG.integers(0,len(blocks),len(blocks))])
               for _ in range(NBOOT)]
        for a,b in [("L2","L0"),("L5","L2")]:
            obs=roc_auc_score(chg,S[a])-roc_auc_score(chg,S[b])
            dl=np.array([roc_auc_score(chg[i],S[a][i])-roc_auc_score(chg[i],S[b][i])
                         for i in picks if 0<chg[i].mean()<1])
            lo,hi=np.percentile(dl,[2.5,97.5])
            rows.append({"horizon_h":h*6,"scope":scope,"contrast":f"{a} - {b}",
                         "delta_auroc":round(float(obs),4),"lo":round(float(lo),4),
                         "hi":round(float(hi),4),
                         "excludes_zero":bool(lo>0 or hi<0),
                         "p_boot":round(float(2*min((dl<=0).mean(),(dl>=0).mean())),4)})
        print(f"  h={h*6:>2}h {scope} done")

R=pd.DataFrame(rows); R.to_csv(PRED/"horizon_contrasts.csv", index=False)
pd.set_option("display.width",170)
for scope in R.scope.unique():
    print(f"\n=== {scope} ===")
    print(R[R.scope==scope][["horizon_h","contrast","delta_auroc","lo","hi","excludes_zero","p_boot"]].to_string(index=False))
print(f"\nwrote {PRED/'horizon_contrasts.csv'}")
