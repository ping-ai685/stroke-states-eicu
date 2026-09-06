"""
Paper 2: check that every number drawn in a figure equals the one printed in a table.

Figures 1, 3 and 4 read the same CSVs the tables read, so agreement there is
structural. Figure 2 does not: it re-decodes MIMIC-IV from the frozen model and
recomputes the per-state correlations and prevalences itself. That makes it the
one panel where a state-label mix-up would show as a plausible-looking picture
rather than as an error -- and the labels there are mapped through `raw_state`,
which is correct only because the labels come from `mdl.predict`, never from a
stored `state_k4` column. Read that way it looks like the bug found in script 29.
The only way to tell the two apart is to recompute and compare.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
# The pickled model refers to the custom emission class by module path, so the
# module has to be importable before torch.load can rebuild it.
sys.path.insert(0, str(ROOT / "03_code"))
from pomegranate_patches import CategoricalMixed          # noqa: F401,E402

ORDER = ["neurologically preserved-low support",
         "neurological impairment-low support",
         "neurological impairment-renal dysfunction",
         "neurological impairment-respiratory support"]
# Paper 1's published state prevalences. Not retyped from memory: imported from
# the constant in script 29, which is asserted against the published table every
# time that script runs. An earlier draft of this file carried hand-entered
# values, three of the four wrong -- a verification script that invents its own
# reference values verifies nothing.
_p29 = (HERE / "29_build_prediction_datasets.py").read_text()
PUBLISHED_MIMIC_PCT = eval(
    _p29[_p29.index("PUBLISHED_PCT = {") + len("PUBLISHED_PCT = "):]
    .split("}")[0] + "}")

fails = []


def cmp(label, drawn, printed, tol):
    ok = abs(drawn - printed) <= tol
    print(f"  {'OK ' if ok else 'BAD'}  {label:52s} 图 {drawn:8.3f}   表 {printed:8.3f}")
    if not ok:
        fails.append(label)


print("Figure 2 —— 重新走一遍作图脚本的取数，与 per_state_full.csv 比对\n")

FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()
lblmap = (pd.read_csv(ROOT / "04_outputs/tables/state_label_mapping.csv")
          .set_index("raw_state")["name"].to_dict())

mdl = torch.load(ROOT / ".model_k4.pt", weights_only=False)
for d in mdl.distributions:
    for s in d.distributions[-4:]:
        p = torch.clamp(s.probs.detach().clone(), 1e-4, 1 - 1e-4)
        p = p / p.sum(-1, keepdim=True)
        s.probs = torch.nn.Parameter(p, requires_grad=False)
        s._log_probs = torch.log(p)

mw = pd.read_csv(ROOT / "04_outputs/tables/timewindow_level_modeling.csv").sort_values(
    ["stay_id", "window_idx"])
mw["state"] = pd.Series(np.concatenate(
    [mdl.predict(torch.tensor(g[FEATS].to_numpy(dtype=np.float32)).unsqueeze(0))[0].numpy()
     for _, g in mw.groupby("stay_id", sort=False)]), index=mw.index).map(lblmap)
MZ = mw.groupby("state")[FEATS].mean()

a = pd.read_csv(HERE / "results_full_model/eicu_state_assignments_full_A1.csv")
mo = pd.read_csv(HERE / "windows/timewindow_level_modeling_eicu_A1.csv")
el = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
bi = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv")
ew = mo.merge(bi, on=["patientunitstayid", "window_idx"]).merge(
    a[["patientunitstayid", "window_idx", "state", "vent_ascertainable"]],
    on=["patientunitstayid", "window_idx"])
sub = ew[ew.patientunitstayid.isin(el) & (ew.vent_ascertainable == 1)]
EZ = sub.groupby("state")[FEATS].mean()

ps = pd.read_csv(HERE / "results_full_model/per_state_full.csv")
psb = ps[ps.scope == "both"].set_index("state")
for c in ["mimic_prev", "eicu_prev", "profile_corr"]:
    assert c in psb.columns, f"per_state_full.csv 少了 {c} 列"

print("上排：每个状态的 profile correlation")
for s in ORDER:
    r = float(np.corrcoef(MZ.loc[s].values, EZ.loc[s].values)[0, 1])
    cmp(s, r, float(psb.loc[s, "profile_corr"]), 0.0015)

print("\n下排：eICU 各状态占比 (%)")
ep = (sub.state.value_counts(normalize=True) * 100).reindex(ORDER)
for s in ORDER:
    cmp(s, float(ep[s]), float(psb.loc[s, "eicu_prev"]), 0.06)

print("\n下排：MIMIC-IV 各状态占比 (%)，同时对照本流程的表与第一篇已发表的值")
mp = (mw.state.value_counts(normalize=True) * 100).reindex(ORDER)
for s in ORDER:
    cmp(s + " (vs 表)", float(mp[s]), float(psb.loc[s, "mimic_prev"]), 0.06)
    cmp(s + " (vs 第一篇)", float(mp[s]), PUBLISHED_MIMIC_PCT[s], 0.06)

# ---------------------------------------------------------------- Figure 1
print("\n\nFigure 1 —— 流程图上的每个格子与排除数\n")
flow = pd.read_csv(HERE / "cohort/cohort_flow_counts.csv")
for i in range(1, len(flow)):
    drawn = int(flow.n_stays[i - 1] - flow.n_stays[i])
    print(f"  {'OK ' if drawn >= 0 else 'BAD'}  第{i}步排除 {drawn:>6d} 例住院  "
          f"({flow.step[i][:44]})")
    if drawn < 0:
        fails.append(f"figure1 step {i}")
cmp("最终患者数", float(flow.n_patients.iloc[-1]), 8279.0, 0.5)
cmp("最终医院数", float(flow.n_hospitals.iloc[-1]), 176.0, 0.5)

# ---------------------------------------------------------------- Figure 3
print("\n\nFigure 3 —— 热图上的相关系数 vs 表里的对应相关\n")
CONT = [f for f in FEATS if f.endswith("_z")]


def emission_means(model, n):
    return np.array([[float(d.distributions[i].means.detach().ravel()[0])
                      for i in range(len(CONT))] for d in model.distributions])


Pm = emission_means(mdl, 4)
for tag, pt, mfile in [("K=3", "denovo_K3_matching.csv", ".model_k3_eicu_denovo.pt"),
                       ("K=4", "denovo_K4_matching.csv", ".model_k4_eicu_denovo.pt")]:
    m = torch.load(HERE / "results_denovo" / mfile, weights_only=False)
    Q = emission_means(m, len(m.distributions))
    C = np.array([[np.corrcoef(Q[j], Pm[k])[0, 1] for k in range(4)]
                  for j in range(len(Q))])
    tb = pd.read_csv(HERE / "results_denovo" / pt)
    col = "best_match_correlation"
    print(f"  {tag}")
    for j, row in tb.iterrows():
        drawn = float(C[j].max())            # the heatmap's best cell for that row
        cmp(f"   eICU state {row.eicu_state} 最佳匹配", drawn, float(row[col]), 0.0015)

# ---------------------------------------------------------------- Figure 4
print("\n\nFigure 4 —— 「保留比例」与表 4 印的值\n")
MH = pd.read_csv(HERE / "prediction/multihorizon_metrics.csv")
L6 = pd.read_csv(HERE / "prediction/metrics_L6.csv")   # L6 lives in its own file
PRINTED = {6: (0.730, 0.787, 80.2), 12: (0.754, 0.799, 84.9),
           24: (0.780, 0.798, 94.1)}                   # Table 4, panel B
for h, (p_l2, p_l6, p_share) in PRINTED.items():
    l2 = float(MH[(MH.scope == "eicu-full") & (MH.horizon_h == h)
                  & (MH.model == "L2")].change_auroc.iloc[0])
    l6 = float(L6[(L6.scope == "eicu-full") & (L6.horizon_h == h)].change_auroc.iloc[0])
    cmp(f"{h} h 冻结模型 AUROC", l2, p_l2, 0.0006)
    cmp(f"{h} h 序列模型 AUROC", l6, p_l6, 0.0006)
    cmp(f"{h} h 保留比例 %", (l2 - .5) / (l6 - .5) * 100, p_share, 0.06)

print()
if fails:
    print(f"❌ 共 {len(fails)} 项对不上：")
    for f in dict.fromkeys(fails):
        print("   ", f)
    raise SystemExit(1)
print("✅ 四张图画出来的每个数都等于表里印的数")
