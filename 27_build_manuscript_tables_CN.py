"""
Paper 2: Chinese manuscript tables.

Reads the same CSVs as 25_build_manuscript_tables.py and emits the same rows with
translated captions and headers, so the two language versions cannot diverge in
their numbers. Row counts are asserted against the English build.
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "manuscript"
P = "neurologically preserved-low support"; R = "neurological impairment-respiratory support"
N = "neurological impairment-renal dysfunction"; L = "neurological impairment-low support"
ORDER = [P, L, N, R]
CN = {P: "神经功能保留–低支持", L: "神经功能受损–低支持",
      N: "神经功能受损–肾功能异常", R: "神经功能受损–呼吸支持"}

def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"]*len(header)) + "|"] +
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])
T = {}

coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
def iqr(s): return f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"
def pct(s): return f"{s.sum():,}（{s.mean()*100:.1f}）"
groups = [("全部患者", coh)] + [(g, coh[coh.stroke_subtype == g]) for g in ["AIS","ICH","SAH","ICH+SAH","unspecified"]]
rows = []
for lab, fn in [("患者数 n", lambda d: f"{len(d):,}"),
                ("年龄，中位数 [IQR]", lambda d: iqr(d.age_n)),
                ("女性 n（%）", lambda d: pct(d.gender == "Female")),
                ("神经 ICU n（%）", lambda d: pct(d.unittype == "Neuro ICU")),
                ("教学医院 n（%）", lambda d: pct(d.teachingstatus == "t")),
                ("ICU 停留，天，中位数 [IQR]", lambda d: f"{d.icu_los_days.median():.2f} [{d.icu_los_days.quantile(.25):.2f}–{d.icu_los_days.quantile(.75):.2f}]"),
                ("贡献 6 小时窗口，中位数", lambda d: f"{d.n_windows.median():.0f}"),
                ("ICU 死亡 n（%）", lambda d: pct(d.icu_mortality == 1)),
                ("院内死亡 n（%）", lambda d: pct(d.hospital_mortality == 1))]:
    rows.append([lab] + [fn(d) for _, d in groups])
T[1] = ("**表 1.** eICU 外部验证队列的特征，总体及按卒中亚型。亚型优先级沿用发现研究；仅携带 unspecified stroke 路径的病例单列一栏，其排除为预设敏感性分析。",
        md(rows, ["特征"] + [g for g, _ in groups]))

av = pd.read_csv(HERE / "audit/availability_vs_mimic.csv")
cap = pd.read_csv(HERE / "review_returned/ctier_capture_check.csv").set_index("variable")
NICE = {"heart_rate":"心率","sbp":"收缩压","map":"平均动脉压","resp_rate":"呼吸频率","spo2":"外周血氧饱和度",
        "temp_c":"体温","gcs_eye":"GCS eye 分量","gcs_motor":"GCS motor 分量","urine_output_ml":"尿量",
        "wbc":"白细胞计数","hemoglobin":"血红蛋白","platelet":"血小板计数","creatinine":"肌酐",
        "bun":"尿素氮","sodium":"钠","potassium":"钾","glucose":"血糖"}
TIER = {**{v:"直接协调" for v in ["heart_rate","sbp","map","resp_rate","spo2","temp_c","wbc","hemoglobin","platelet","creatinine","bun","sodium","potassium","glucose"]},
        **{v:"需要验证" for v in ["gcs_eye","gcs_motor","urine_output_ml"]}}
rows = [[NICE[r.variable], TIER[r.variable], f"{r.mimic_pct_windows:.1f}", f"{r.eicu_pct_windows:.1f}", f"{r.difference:+.1f}"]
        for _, r in av.iterrows()]
for name, m, e, d in [("有创机械通气","24.8","17.2–37.0","以区间报告"),
                      ("镇静（持续输注）","19.7", f"{cap.loc['sedative','eicu_pct_windows']:.1f}", f"{cap.loc['sedative','eicu_pct_windows']-19.7:+.1f}"),
                      ("Vasoactive support（持续输注）","9.3", f"{cap.loc['vasopressor','eicu_pct_windows']:.1f}", f"{cap.loc['vasopressor','eicu_pct_windows']-9.3:+.1f}"),
                      ("连续肾替代治疗","0.2", f"{cap.loc['crrt','eicu_pct_windows']:.1f}", f"{cap.loc['crrt','eicu_pct_windows']-0.2:+.1f}")]:
    rows.append([name, "需要算法重建", m, e, d])
T[2] = ("**表 2.** 21 个模型变量的窗口级可得率，在两个数据库中以完全相同的口径、于任何前向填充之前计算。有创通气以区间报告，因为没有任何 eICU 医院能对超过 67% 的卒中患者判定有创与否；其下界不应与发现值作字面比较。",
        md(rows, ["变量","协调层级","MIMIC-IV，窗口占比 %","eICU，窗口占比 %","差值"]))

tm = pd.read_csv(HERE / "results_full_model/transport_metrics_full.csv")
SC = {"full":"全队列","GCS-eligible":"GCS 合格医院","vent-ascertainable":"通气可确定医院","both":"同时满足两条规则"}
rows = [[SC[r.scope], f"{r.patients:,}", f"{r.windows:,}", f"{r.min_profile_corr:.3f}", f"{r.min_prevalence_pct:.1f}",
         f"{r.transition_rank_corr:.3f}", f"{r.max_self_transition_diff:.3f}", f"{r.null_p95:.3f}", f"{r.pct_changing_state:.1f}"]
        for _, r in tm.iterrows()]
T[3] = ("**表 3.** 冻结模型迁移对照五条预设标准。阈值：per-state profile correlation ≥0.80；每个状态 ≥5% 窗口；transition rank correlation ≥0.80；最大自转换偏差 ≤0.10；观测最低 profile correlation 高于置换零分布第 95 百分位。五条在四个范围内全部达成。变化状态的患者比例为描述性指标，不参与通过／未通过判定；发现值为 40.3%。",
        md(rows, ["分析范围","患者","窗口","最低 profile r","最小构成 %","Transition rank r","最大自转换 Δ","零分布 95 分位","变化状态 %"]))

oa = pd.read_csv(HERE / "results_full_model/outcome_association_full.csv")
oa = oa[(oa.model=="adjusted") & (oa.scope=="full")]
MI = {("new invasive ventilation within 12 h",L):"1.64（1.14–2.35）",("new invasive ventilation within 12 h",N):"3.25（2.32–4.55）",
      ("new invasive ventilation within 12 h",R):"4.28（3.10–5.92）",("new vasoactive support within 12 h",L):"1.21（0.81–1.79）",
      ("new vasoactive support within 12 h",N):"4.91（3.79–6.36）",("new vasoactive support within 12 h",R):"5.00（4.07–6.13）",
      ("ICU death (discrete-time hazard)",L):"0.21（0.05–0.87）",("ICU death (discrete-time hazard)",N):"3.73（2.48–5.59）",
      ("ICU death (discrete-time hazard)",R):"11.83（9.00–15.55）"}
OCN = {"new invasive ventilation within 12 h":"12 小时内新发有创通气","new vasoactive support within 12 h":"12 小时内新发 vasoactive support",
       "ICU death (discrete-time hazard)":"ICU 死亡（离散时间风险）"}
rows = []
for oc in OCN:
    for st in [L, N, R]:
        row = oa[(oa.outcome==oc)&(oa.state==st)]
        if not len(row): continue
        r0 = row.iloc[0]
        e = f"{r0.OR:.2f}（{r0.CI_low:.2f}–{r0.CI_high:.2f}）" + ("†" if r0.suppressed_lt10_events else "")
        rows.append([OCN[oc], CN[st], MI[(oc,st)], e, int(r0.n_events_in_state)])
T[5] = ("**表 5.** 当前状态与后续事件的校正比值比，参照态为神经功能保留–低支持。采用交换式工作相关结构、按患者聚类的 generalized estimating equations，校正年龄、性别与卒中亚型。† 依少于十个事件的预设规则予以抑制；发现研究对同一状态的估计同样基于两个事件。九项比较全部重现效应方向。",
        md(rows, ["结局","状态","MIMIC-IV OR（95% CI）","eICU OR（95% CI）","eICU 事件数"]))

a3 = pd.read_csv(HERE / "results_aim3/between_hospital_variance.csv").set_index("state")
rows = [[CN[s], f"{a3.loc[s,'pooled_pct']:.1f}", f"{a3.loc[s,'hospital_min_pct']:.1f}–{a3.loc[s,'hospital_max_pct']:.1f}",
         f"{a3.loc[s,'hospital_median_pct']:.1f}", f"{a3.loc[s,'between_hospital_SD_pp']:.1f}", f"{a3.loc[s,'ICC']:.3f}"] for s in ORDER]
T[6] = ("**表 6.** 同时满足两条准入规则的 38 家医院中状态构成的院间变异。方差分量来自随机截距线性概率模型，故标准差可直接读作构成的百分点。",
        md(rows, ["状态","合计 %","院级范围 %","院级中位数 %","院间 SD，百分点","ICC"]))

gs = pd.read_csv(HERE / "results_full_model/gcs_observed_vs_imputed_prevalence.csv")
gc = pd.read_csv(HERE / "results_full_model/gcs_observed_only_profile_corr.csv")
rows = []
for sc, lab in [("full","全队列"),("both-eligible","同时满足两条规则")]:
    for s in ORDER:
        g = gs[(gs.scope==sc)&(gs.state==s)].iloc[0]; c = gc[(gc.scope==sc)&(gc.state==s)].iloc[0]
        rows.append([lab, CN[s], f"{g.mimic_pct:.1f}", f"{g.eicu_observed_gcs_pct:.1f}", f"{g.eicu_imputed_gcs_pct:.1f}",
                     f"{c.corr_all_windows:.3f}", f"{c.corr_observed_gcs_only:.3f}"])
T[4] = ("**表 4.** 迁移对 GCS 填充的敏感性。由于发现研究对 motor 分量的填充中位数与保留态的点质量发射精确重合，状态构成按分量为实测与填充的窗口分别报告，profile correlation 亦在两个数据库同时限制到实测分量窗口后重算。限制到实测窗口使每一个相关性都上升，故填充在削弱迁移而非产生它。",
        md(rows, ["范围","状态","MIMIC-IV 构成 %","eICU 实测 GCS %","eICU 填充 GCS %","Profile r 全部窗口","Profile r 仅实测"]))

en = (OUT / "tables_v1.md").read_text()
blocks = []
for k in sorted(T):
    c, b = T[k]; blocks.append(f"{c}\n\n{b}\n")
    def datarows(block):
        return len([ln for ln in block.splitlines()
                    if ln.startswith("|") and not set(ln) <= set("|- ")]) - 1
    n_cn = datarows(b)
    n_en = datarows(en.split(f"**Table {k}.**")[1].split("**Table ")[0])
    assert n_cn == n_en, f"Table {k} data rows differ: CN {n_cn} vs EN {n_en}"
(OUT / "tables_v1_CN.md").write_text("# 表\n\n" + "\n\n".join(blocks))
print(f"wrote {OUT/'tables_v1_CN.md'} — all 6 tables match the English row counts")
