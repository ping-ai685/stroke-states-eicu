"""
Paper 2, Aim 4: a sequence model, to set the ceiling with a competent opponent.

The claim that the frozen representation reaches most of the achievable
discrimination is only as strong as the best alternative it is compared with.
Steps 30 to 34 used multinomial logistic regression and gradient boosting, both
of which see one window at a time. A reviewer will reasonably ask what a model
that reads the raw observation sequence would do, and the honest answer has to
be measured rather than asserted.

L6 is a GRU over the 21 model variables, with age, sex and stroke subtype
appended at every step, predicting the state h windows ahead from the hidden
state at window t. It therefore sees strictly more than L3-L5 (which get one
window plus the decoded state) and it reaches the observation history without
passing through the frozen model, as L2 does.

Discipline is unchanged: fitted on the MIMIC-IV training split alone, with an
internal split of that training set for early stopping, then frozen before eICU
is decoded. Several seeds are run and reported, because a single seed of a
neural network is not a ceiling.

Output: prediction/sequence_model_metrics.csv, prediction/sequence_contrasts.csv
"""
import copy, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent; ROOT = HERE.parent
T1 = ROOT / "04_outputs/tables"; PRED = HERE / "prediction"
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401

EPS = 1e-4
HORIZONS = [1, 2, 4]
SEEDS = [0, 1, 2]
NBOOT = 300
HIDDEN, LAYERS, EPOCHS, PATIENCE, LR = 64, 1, 60, 8, 3e-3
FEATS = pd.read_csv(HERE/"frozen_params/feature_order.csv").feature.tolist()
STATES = pd.read_csv(T1/"state_label_mapping.csv").set_index("raw_state")["name"].reindex(range(4)).tolist()
SUBTYPES = ["AIS", "ICH", "SAH", "ICH+SAH"]

# ---- frozen model, for the L2 comparison -------------------------------------
model = torch.load(ROOT/".model_k4.pt", weights_only=False)
fl = copy.deepcopy(model)
for d_ in fl.distributions:
    for sub in d_.distributions[-4:]:
        p = torch.clamp(sub.probs.detach().clone(), EPS, 1-EPS); p = p/p.sum(-1, keepdim=True)
        sub.probs = torch.nn.Parameter(p, requires_grad=False); sub._log_probs = torch.log(p)
TRANS = torch.exp(fl.edges).detach().numpy(); TRANS = TRANS/TRANS.sum(1, keepdims=True)

# ---- data --------------------------------------------------------------------
mm = pd.read_csv(PRED/"mimic_pairs.csv"); ee = pd.read_csv(PRED/"eicu_pairs.csv")
mc = pd.read_csv(T1/"patient_level_cohort.csv")[["stay_id","age_at_adm_capped","gender","stroke_subtype"]]
mc = mc.rename(columns={"stay_id":"uid","age_at_adm_capped":"age"}); mc["female"]=(mc.gender=="F").astype(float)
mm = mm.merge(mc[["uid","age","female","stroke_subtype"]], on="uid", how="left")
ec = pd.read_csv(HERE/"cohort/patient_level_cohort.csv")[["patientunitstayid","age_n","gender","stroke_subtype"]]
ec = ec.rename(columns={"patientunitstayid":"uid","age_n":"age"})
ec["female"]=np.where(ec.gender=="Female",1.0,np.where(ec.gender=="Male",0.0,np.nan))
ee = ee.merge(ec[["uid","age","female","stroke_subtype"]], on="uid", how="left")

AGE_M, AGE_S = mm.age.mean(), mm.age.std()          # frozen from MIMIC, like everything else
for d in (mm, ee):
    d["age_z"] = (d.age - AGE_M) / AGE_S
    d["age_z"] = d.age_z.fillna(0.0)
    d["female"] = d.female.fillna(0.5)
    for s in SUBTYPES:
        d[f"sub_{s}"] = (d.stroke_subtype == s).astype(float)
COV = ["age_z", "female"] + [f"sub_{s}" for s in SUBTYPES]
INPUT = FEATS + COV
for d, lab in [(mm, "MIMIC"), (ee, "eICU")]:
    assert d[INPUT].notna().all().all(), f"{lab}: NaN in model input"


def sequences(df):
    """Per-patient padded tensors, plus the state label at every window."""
    df = df.sort_values(["uid", "window_idx"])
    uids, X, Y, L = [], [], [], []
    for uid, g in df.groupby("uid", sort=False):
        uids.append(uid)
        X.append(torch.tensor(g[INPUT].to_numpy(dtype=np.float32)))
        Y.append(torch.tensor(pd.Categorical(g.state, categories=STATES).codes.astype(np.int64)))
        L.append(len(g))
    Xp = nn.utils.rnn.pad_sequence(X, batch_first=True)
    Yp = nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=-1)
    return uids, Xp, Yp, torch.tensor(L)


class GRUNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.gru = nn.GRU(d_in, HIDDEN, LAYERS, batch_first=True)
        self.out = nn.Linear(HIDDEN, 4)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                                   enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True,
                                                total_length=x.shape[1])
        return self.out(h)                       # (B, T, 4) logits at every window


def shift_target(Y, L, h):
    """Target at t+h; -1 where the horizon runs past the end of the stay."""
    T_ = Y.shape[1]
    tgt = torch.full_like(Y, -1)
    if h < T_:
        tgt[:, :T_-h] = Y[:, h:]
    idx = torch.arange(T_).unsqueeze(0)
    tgt[idx >= (L.unsqueeze(1) - h)] = -1
    return tgt


mu, Xm, Ym, Lm = sequences(mm)
eu, Xe, Ye, Le = sequences(ee)
split = mm.groupby("uid").split.first().reindex(mu).to_numpy()
tr_i = np.flatnonzero(split == "train"); te_i = np.flatnonzero(split == "test")
print(f"MIMIC sequences: {len(tr_i):,} train, {len(te_i):,} test | eICU: {len(eu):,}")


def train_one(h, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    perm = torch.randperm(len(tr_i), generator=g).numpy()
    cut = int(0.85 * len(perm))
    fit_i, val_i = tr_i[perm[:cut]], tr_i[perm[cut:]]
    net = GRUNet(len(INPUT))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    Tf = shift_target(Ym, Lm, h)
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        net.train()
        order = torch.randperm(len(fit_i), generator=g)
        for b in range(0, len(fit_i), 256):
            j = fit_i[order[b:b+256].numpy()]
            opt.zero_grad()
            logit = net(Xm[j], Lm[j])
            loss = lossf(logit.reshape(-1, 4), Tf[j].reshape(-1))
            if torch.isfinite(loss):
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xm[val_i], Lm[val_i]).reshape(-1, 4), Tf[val_i].reshape(-1)).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state); net.eval()
    return net, best


def predict(net, X, L, idx, h):
    with torch.no_grad():
        logit = net(X[idx], L[idx])
    P = torch.softmax(logit, dim=2).numpy()
    keep, rows = shift_target(Y_of[id(X)], L, h)[idx].numpy() >= 0, []
    return P, keep


Y_of = {id(Xm): Ym, id(Xe): Ye}


def filtered(df):
    df = df.sort_values(["uid", "window_idx"]); out = np.full((len(df), 4), np.nan)
    pos = {k: i for i, k in enumerate(df.index)}; by = {}
    for uid, g in df.groupby("uid", sort=False): by.setdefault(len(g), []).append((uid, g))
    for Lg, items in sorted(by.items()):
        Xb = np.stack([g[FEATS].to_numpy(dtype=np.float32) for _, g in items])
        with torch.no_grad(): a = fl.forward(torch.tensor(Xb)).numpy()
        a = a - a.max(2, keepdims=True); f = np.exp(a); f = f / f.sum(2, keepdims=True)
        for k, (_, g) in enumerate(items):
            for j, ix in enumerate(g.index): out[pos[ix]] = f[k, j]
    return pd.DataFrame(out, index=df.index, columns=[f"f|{s}" for s in STATES])

print("forward filtering for the L2 comparison...")
mmf = mm.reset_index(drop=True); eef = ee.reset_index(drop=True)
mmf = mmf.join(filtered(mmf)); eef = eef.join(filtered(eef))
FC = [f"f|{s}" for s in STATES]

rows, crows = [], []
RNG = np.random.default_rng(42)
for h in HORIZONS:
    Th = np.linalg.matrix_power(TRANS, h)
    nets = []
    for sd in SEEDS:
        net, vl = train_one(h, sd)
        nets.append(net)
        print(f"  h={h*6:>2}h seed {sd}: val loss {vl:.4f}")

    for scope, X, L, idx, flat in [
            ("mimic-test", Xm, Lm, te_i, mmf[mmf.split == "test"]),
            ("eicu-full", Xe, Le, np.arange(len(eu)), eef)]:
        Yb = Y_of[id(X)]
        tgt = shift_target(Yb, L, h)[idx].numpy()
        cur = Yb[idx].numpy()
        mask = tgt >= 0
        chg = (tgt != cur)[mask].astype(int)
        # One uid per (patient, window) cell; the same mask the scores carry.
        uids_arr = np.array(eu if scope == "eicu-full" else [mu[i] for i in te_i])
        # frozen model on exactly the same rows
        d = flat.sort_values(["uid", "window_idx"])
        F = np.zeros((len(idx), X.shape[1], 4))
        p = 0
        for r, n in enumerate(L[idx].numpy()):
            F[r, :n] = d[FC].to_numpy()[p:p+n]; p += n
        assert p == len(d), (p, len(d))
        L2 = (F @ Th)
        L2 = L2 / np.clip(L2.sum(2, keepdims=True), 1e-12, None)
        s2 = np.round(1.0 - L2[np.arange(len(idx))[:, None], np.arange(X.shape[1]), cur], 12)[mask]

        s6, Pk = [], []
        for net in nets:
            with torch.no_grad():
                P = torch.softmax(net(X[idx], L[idx]), dim=2).numpy()
            Pk.append(P)
            s6.append(np.round(1.0 - P[np.arange(len(idx))[:, None], np.arange(X.shape[1]), cur], 12)[mask])
        for sd, sc in zip(SEEDS, s6):
            rows.append({"horizon_h": h*6, "scope": scope, "seed": sd, "n": int(mask.sum()),
                         "changed_pct": round(float(chg.mean()*100), 2),
                         "L6_auroc": round(float(roc_auc_score(chg, sc)), 4),
                         "L2_auroc": round(float(roc_auc_score(chg, s2)), 4)})
        s6m = np.mean(s6, axis=0)                       # seed-averaged predictor
        # Persist the seed-averaged distribution so the full metric set and the
        # manuscript tables can be computed without retraining.
        Pm = np.mean(Pk, axis=0)
        flat_uid = np.repeat(uids_arr[:, None], X.shape[1], axis=1)[mask]
        flat_widx = np.repeat(np.arange(X.shape[1])[None, :], len(idx), axis=0)[mask]
        out = pd.DataFrame({"uid": flat_uid, "window_idx": flat_widx,
                            "state": np.array(STATES)[cur[mask]],
                            "target": np.array(STATES)[tgt[mask]]})
        for j, st in enumerate(STATES):
            out[f"L6|{st}"] = Pm[mask][:, j]
        out.to_csv(PRED / f"pred_probs_L6_{scope}_h{h*6}.csv", index=False)
        uid_masked = np.repeat(uids_arr[:, None], X.shape[1], axis=1)[mask]
        assert len(uid_masked) == len(chg), (len(uid_masked), len(chg))
        _, inv = np.unique(uid_masked, return_inverse=True)
        blocks = [np.flatnonzero(inv == i) for i in range(inv.max()+1)]
        obs = roc_auc_score(chg, s6m) - roc_auc_score(chg, s2)
        dl = []
        for _ in range(NBOOT):
            b = np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))])
            if 0 < chg[b].mean() < 1:
                dl.append(roc_auc_score(chg[b], s6m[b]) - roc_auc_score(chg[b], s2[b]))
        lo, hi = np.percentile(dl, [2.5, 97.5])
        crows.append({"horizon_h": h*6, "scope": scope, "contrast": "L6 - L2",
                      "L6_auroc_seedmean": round(float(roc_auc_score(chg, s6m)), 4),
                      "L2_auroc": round(float(roc_auc_score(chg, s2)), 4),
                      "delta_auroc": round(float(obs), 4),
                      "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                      "excludes_zero": bool(lo > 0 or hi < 0)})
        print(f"    {scope:<11} n={int(mask.sum()):>6,}  L6 {roc_auc_score(chg,s6m):.4f}  "
              f"L2 {roc_auc_score(chg,s2):.4f}  delta {obs:+.4f} ({lo:+.4f},{hi:+.4f})")

pd.DataFrame(rows).to_csv(PRED/"sequence_model_metrics.csv", index=False)
C = pd.DataFrame(crows); C.to_csv(PRED/"sequence_contrasts.csv", index=False)
print("\n=== L6 (GRU, seed-averaged) versus L2 (frozen HMM) ===")
pd.set_option("display.width", 170)
print(C.to_string(index=False))
