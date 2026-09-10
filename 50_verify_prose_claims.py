"""
Paper 2: check the claims that are not numbers.

Scripts 23, 47 and 49 all check digits. But the sentences a reviewer will test
first are the ones with no digit in them at all -- "all nine reproduced the
direction", "the ordering is identical throughout", "correlations above 0.97".
Those were verified by reading a table, once, by one person. Each one below is
restated as something a computer can fail.
"""
import re

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
R = []


def claim(text, ok, detail=""):
    R.append((bool(ok), text, detail))
    print(f"  {'OK ' if ok else 'BAD'}  {text}")
    if detail:
        print(f"         {detail}")


print("Results / Abstract\n")

# "All five transportability criteria were met in all four scopes."
tm = pd.read_csv(HERE / "results_full_model/transport_metrics_full.csv")
met = ((tm.min_profile_corr >= 0.80) & (tm.min_prevalence_pct >= 5.0)
       & (tm.transition_rank_corr >= 0.80) & (tm.max_self_transition_diff <= 0.10)
       & (tm.min_profile_corr > tm.null_p95))
claim("五条迁移标准在四个范围内全部达成",
      len(tm) == 4 and met.all(),
      f"{int(met.sum())}/{len(tm)} 个范围通过全部五条")

# "the 38 of the 176 hospitals that satisfy both eligibility rules". Nothing else
# guards the hospital counts, and a reader will recompute them from the scopes.
_a = pd.read_csv(HERE / "results_full_model/eicu_state_assignments_full_A1.csv")
_e = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
_H = {"full": _a,
      "GCS-eligible": _a[_a.patientunitstayid.isin(_e)],
      "vent-ascertainable": _a[_a.vent_ascertainable == 1],
      "both": _a[_a.patientunitstayid.isin(_e) & (_a.vent_ascertainable == 1)]}
_n = {k: v.hospitalid.nunique() for k, v in _H.items()}
claim("全队列 176 家医院，两条规则都满足的是 38 家",
      _n["full"] == 176 and _n["both"] == 38,
      "  ".join(f"{k} {v}" for k, v in _n.items()))
# The de novo refit used the strictest scope, not the full 176-hospital cohort.
# Reading the manuscript, one would have assumed the full cohort: neither Methods
# nor Results said otherwise until this was checked.
_dn = pd.read_csv(HERE / "results_denovo/eicu_denovo_assignments.csv")
_a2 = pd.read_csv(HERE / "results_full_model/eicu_state_assignments_full_A1.csv")
_dnh = _a2[_a2.patientunitstayid.isin(set(_dn.patientunitstayid))].hospitalid.nunique()
claim("从零拟合用的是 38 家医院 4296 名患者，不是全队列",
      _dnh == 38 and _dn.patientunitstayid.nunique() == 4296,
      f"医院 {_dnh}，患者 {_dn.patientunitstayid.nunique():,}，窗口 {len(_dn):,}")
for _f, _w in [("manuscript/methods_jamia_EN.md", "in the strictest scope"),
               ("manuscript/results_jamia_EN.md", "the 38 hospitals"),
               ("manuscript/abstract_jamia_EN.md", "de novo in the strictest scope"),
               ("manuscript/discussion_jamia_EN.md", "within those same 38 hospitals"),
               ("manuscript/cover_letter_JAMIA.md", "within the 38 hospitals")]:
    claim(f"{_f.split('/')[-1]} 交代了从零拟合的范围",
          _w in (HERE / _f).read_text(), f"应含「{_w}」")

_rj = (HERE / "manuscript/results_jamia_EN.md").read_text()
claim("结果部分说明了这一项分析为何只用 38 家",
      "38 of the 176 hospitals" in _rj and "the other analyses use the full cohort" in _rj)

# "All nine state-outcome comparisons reproduced the direction of association."
vs = pd.read_csv(HERE / "results_full_model/outcome_association_vs_mimic.csv")
same = (vs.MIMIC_OR - 1).apply(np.sign) == (vs.eICU_OR - 1).apply(np.sign)
claim("九项状态×结局比较全部重现关联方向",
      len(vs) == 9 and same.all(),
      f"{int(same.sum())}/{len(vs)} 项方向一致（按 OR 相对 1 的正负重算，未用 CSV 自带的 direction_same 列）")

# "with exact rank order for invasive ventilation and ICU death"
for out in vs.outcome.unique():
    q = vs[vs.outcome == out]
    ok = list(q.sort_values("MIMIC_OR").state) == list(q.sort_values("eICU_OR").state)
    if "ventilation" in out or "death" in out:
        claim(f"次序完全保留：{out}", ok,
              "MIMIC 次序 " + " < ".join(q.sort_values("MIMIC_OR").state.str[-14:]))

# "States persisted across 90.7% of pairs" / "a no-change rule scores 0.906"
m = pd.read_csv(HERE / "prediction/metrics_by_model.csv")
ef = m[(m.scope == "eicu-full") & (m.model == "L2")].iloc[0]
claim("90.7% 的窗口配对状态不变", abs((100 - ef.changed_pct) - 90.7) < 0.05,
      f"1 − changed_pct = {100 - ef.changed_pct:.2f}%")
claim("不变规则准确率 0.906", abs((100 - ef.changed_pct) / 100 - 0.906) < 0.0006)

# Results now quotes sensitivity at all three horizons for both models. These are
# prose numbers, so nothing else guards them.
_rjE = (HERE / "manuscript/results_jamia_EN.md").read_text()
_MH2 = pd.read_csv(HERE / "prediction/multihorizon_metrics.csv")
_L62 = pd.read_csv(HERE / "prediction/metrics_L6.csv")
_fz, _sq = [], []
for _h in [6, 12, 24]:
    _fz.append(f"{float(_MH2[(_MH2.scope=='eicu-full')&(_MH2.horizon_h==_h)&(_MH2.model=='L2')].change_sens_at_90spec.iloc[0])*100:.1f}")
    _sq.append(f"{float(_L62[(_L62.scope=='eicu-full')&(_L62.horizon_h==_h)].change_sens_at_90spec.iloc[0])*100:.1f}")
claim("结果正文引用的六个灵敏度与结果文件一致",
      all(f"{v}%" in _rjE or v in _rjE for v in _fz + _sq),
      f"冻结 {_fz}，序列 {_sq}")
# The AUROC gap the Results now states before the retained-discrimination ratio.
_g6 = float(_L62[(_L62.scope=='eicu-full')&(_L62.horizon_h==6)].change_auroc.iloc[0]) - \
      float(_MH2[(_MH2.scope=='eicu-full')&(_MH2.horizon_h==6)&(_MH2.model=='L2')].change_auroc.iloc[0])
_g24 = float(_L62[(_L62.scope=='eicu-full')&(_L62.horizon_h==24)].change_auroc.iloc[0]) - \
       float(_MH2[(_MH2.scope=='eicu-full')&(_MH2.horizon_h==24)&(_MH2.model=='L2')].change_auroc.iloc[0])
claim("结果正文的 AUROC 差距 0.057 → 0.018 与数据一致",
      abs(_g6 - 0.057) < 0.0006 and abs(_g24 - 0.018) < 0.0006
      and "from 0.057 to 0.018" in _rjE,
      f"6h 差 {_g6:.4f}，24h 差 {_g24:.4f}")

# medRxiv requires an ethics declaration at submission, and most journals expect
# one even for public de-identified data. Paper 2 had none until this was checked.
# Every factual element below is sourced: the BIDMC waiver from the MIMIC-IV
# release, the Safe Harbor certification from the eICU-CRD PhysioNet record.
for _lang, _f, _must in [
    ("EN", "declarations_EN.md",
     ["Institutional Review Board of Beth Israel Deaconess Medical Center",
      "Safe Harbor", "1031219-2", "Data or Specimens Only",
      "no additional institutional review board approval"]),
    ("CN", "declarations_CN.md",
     ["Beth Israel Deaconess Medical Center", "Safe Harbor", "1031219-2",
      "Data or Specimens Only", "无需追加机构审查委员会批准"]),
]:
    _t = re.sub(r"\s+", " ", (HERE / "manuscript" / _f).read_text())
    _miss = [w for w in _must if w not in _t]
    claim(f"{_lang} 伦理声明覆盖两个数据库且每项都有出处", not _miss, f"缺 {_miss}" if _miss else "")

# JAMIA submission requirements, taken from the journal's General Instructions.
# These are not claims about the data; they are conditions the file must meet
# before it can be uploaded, and nothing else here was checking them.
_tp = (HERE / "manuscript/title_page_EN.md").read_text()
_kw = [k.strip() for k in
       re.search(r"\*\*Keywords:\*\*([^\n]*)", _tp).group(1).split(";") if k.strip()]
claim(f"关键词不超过 5 个（JAMIA: up to five）", len(_kw) <= 5, f"现有 {len(_kw)} 个")

# JAMIA asks for "postal address, e-mail and telephone number" on the title page.
# The authors decided not to print a telephone number; submission portals collect
# the corresponding author's phone in the submission form itself. What must not
# happen is shipping an unfinished placeholder, which is what this now checks.
claim("扉页没有残留的占位符",
      "TO BE SUPPLIED" not in _tp and "待补" not in
      (HERE / "manuscript/title_page_CN.md").read_text(),
      "通讯作者电话按作者决定不在扉页印出，由投稿系统表单提供")

_fl = (HERE / "manuscript/figure_legends_jamia_EN.md").read_text()
_blocks = [b for b in _fl.split("\n\n") if b.strip()]
_ok = True
for i in range(len(_blocks)):
    if _blocks[i].strip().startswith("**Figure "):
        nxt = _blocks[i + 1].strip() if i + 1 < len(_blocks) else ""
        if not nxt.startswith("Alt text:"):
            _ok = False
claim("四张图的 alt text 紧接在各自图注之下，以「Alt text:」开头",
      _ok and _fl.count("Alt text:") == 4, f"共 {_fl.count('Alt text:')} 条")

print("\nSupplementary Table S4\n")

# "Every difference excludes zero in both databases and in all four eICU scopes."
pc = pd.read_csv(HERE / "prediction/prediction_contrasts.csv")
lo = [c for c in pc.columns if c.endswith("_lo") or c == "lo"][0]
hi = [c for c in pc.columns if c.endswith("_hi") or c == "hi"][0]
d = [c for c in pc.columns if "diff" in c or "delta" in c][0]
excl = ~((pc[lo] <= 0) & (pc[hi] >= 0))
claim("每一个相邻梯级之差都不含 0", excl.all(),
      f"{int(excl.sum())}/{len(pc)} 个对比区间不跨 0；数据集 {sorted(pc.scope.unique())}")

# "the ordering of predictors is identical throughout"
MHo = pd.read_csv(HERE / "prediction/multihorizon_metrics.csv")
orders = {}
for (sc, h), g in MHo.groupby(["scope", "horizon_h"]):
    orders[(sc, h)] = tuple(g.sort_values("change_auroc").model)
uniq = set(orders.values())
claim("预测器的排序在所有范围与跨度上完全一致", len(uniq) == 1,
      f"出现 {len(uniq)} 种排序；" + ("一致为 " + " < ".join(next(iter(uniq))) if len(uniq) == 1
                                       else str(sorted(uniq))[:200]))

# "Discrimination rises rather than decays."
rise = all(list(g.sort_values("horizon_h").change_auroc) ==
           sorted(g.sort_values("horizon_h").change_auroc)
           for _, g in MHo[MHo.model == "L2"].groupby("scope"))
claim("跨度拉长时判别力上升而非衰减（冻结模型 L2）", rise)

# Table 4, panel B: the new sensitivity column, and the two values the caption
# quotes from it. Caption and cell are generated from one string in script 42, but
# nothing stops a later edit from retyping one of them.
tj = (HERE / "manuscript/tables_jamia.md").read_text()
_MH = pd.read_csv(HERE / "prediction/multihorizon_metrics.csv")
_L6 = pd.read_csv(HERE / "prediction/metrics_L6.csv")
for h in [6, 12, 24]:
    a = float(_MH[(_MH.scope == "eicu-full") & (_MH.horizon_h == h)
                  & (_MH.model == "L2")].change_sens_at_90spec.iloc[0]) * 100
    b = float(_L6[(_L6.scope == "eicu-full")
                  & (_L6.horizon_h == h)].change_sens_at_90spec.iloc[0]) * 100
    cell = f"{a:.1f} / {b:.1f}"
    claim(f"表 4 面板 B：{h} 小时灵敏度格「{cell}」与结果文件一致", cell in tj)
_s6 = f"{float(_MH[(_MH.scope=='eicu-full')&(_MH.horizon_h==6)&(_MH.model=='L2')].change_sens_at_90spec.iloc[0])*100:.1f}"
_s24 = f"{float(_MH[(_MH.scope=='eicu-full')&(_MH.horizon_h==24)&(_MH.model=='L2')].change_sens_at_90spec.iloc[0])*100:.1f}"
claim("表 4 图注引用的两个灵敏度与表格单元一致",
      f"identifies {_s6}% of transitions at six hours" in tj and f"{_s24}% at" in tj,
      f"图注应为 {_s6}% 与 {_s24}%")

# Both preprints must be disclosed with their real DOIs: Paper 1's, and this
# manuscript's own, which existed only as "at the time of this submission" until
# medRxiv posted it.
_clr = re.sub(r"\s+", " ", (HERE / "manuscript/cover_letter_JAMIA.md").read_text())
for _label, _doi in [("Paper 1", "10.64898/2026.08.30.26361738"),
                     ("this manuscript", "10.64898/2026.09.07.26362407")]:
    claim(f"投稿信披露了{_label}的预印本 DOI", _doi in _clr, _doi)
claim("投稿信不再写「投稿时同步发布」",
      "at the time of this submission" not in _clr,
      "DOI 已知，应写具体号")

# The cover letter carries its own AI disclosure, and an editor reads it beside the
# manuscript's. An earlier version said the tools contributed nothing to "scientific
# conclusions" -- the same overstatement that was corrected in the manuscript and
# not propagated here.
_cl = re.sub(r"\s+", " ", (HERE / "manuscript/cover_letter_JAMIA.md").read_text())
claim("投稿信的 AI 声明与正稿一致",
      "including passages that interpret the findings" in _cl
      and "not used for code, study design or analysis" in _cl
      and "or scientific conclusions" not in _cl,
      "两份声明必须说同一件事，编辑会并排看")

# Internal consistency between two statements that are written far apart and are
# read together by exactly one person: a reviewer. The AI disclosure says Claude
# executed analyses at the authors' direction; Author contributions therefore
# cannot also say the first author "performed all analyses". A reader who notices
# the pair reads it as one of the two being untrue.
for _lang, _f, _exec, _forbidden, _required in [
    ("EN", "declarations_EN.md", "to execute analyses at the authors' direction",
     ["performed all analyses"], "directed and verified all analyses"),
    ("CN", "declarations_CN.md", "在作者指示下运行分析",
     ["完成全部分析"], "主导并核验全部分析"),
]:
    # Source files are hand-wrapped, so a phrase can straddle a line break.
    # Compare on whitespace-normalised text, not on the file's line layout.
    _t = re.sub(r"\s+", " ", (HERE / "manuscript" / _f).read_text())
    if _exec in _t:
        _clash = [w for w in _forbidden if w in _t]
        claim(f"{_lang} 作者贡献与 AI 声明不矛盾（AI 执行分析 → 作者「主导并核验」）",
              not _clash and _required in _t,
              (f"与声明冲突：{_clash}" if _clash else "") +
              ("" if _required in _t else f" 缺「{_required}」"))

# Wording, not arithmetic. "独立可重发现性" stacks 可+重+发现+性 into a
# nominalisation Chinese does not form, and a reader has to guess where it splits.
# The Chinese uses verb phrases instead. Nothing else in this suite checks wording,
# and this is the class of defect the numeric checks cannot see.
_BAD_CN = ["可重发现性", "独立可重发现"]
for _f in ["title_page_CN.md", "abstract_jamia_CN.md", "intro_jamia_CN.md",
           "methods_jamia_CN.md", "results_jamia_CN.md", "discussion_jamia_CN.md",
           "tables_jamia_CN.md", "figure_legends_jamia_CN.md"]:
    _p = HERE / "manuscript" / _f
    if not _p.exists():
        continue
    _t = _p.read_text()
    _hit = [w for w in _BAD_CN if w in _t]
    claim(f"{_f} 未使用生造的「可重发现性」", not _hit, f"命中 {_hit}" if _hit else "")

# The AI disclosure was settled by the authors, twice: it must state that Claude
# drafted interpretive passages (an earlier version denied this), that ChatGPT
# touched no code or design, and it must not claim the authors corrected errors --
# a clause the authors decided against. Wording, not arithmetic; nothing else here
# would notice it drifting back.
for _lang, _f, _must, _mustnt in [
    ("EN", "declarations_EN.md",
     ["including passages that interpret the findings",
      "not used for code, study design or analysis",
      "Neither tool is listed as an author"],
     ["corrected errors introduced"]),
    ("CN", "declarations_CN.md",
     ["包括对结果作出解读的段落", "未用于代码、研究设计或分析", "均未列为作者"],
     ["纠正了其间出现的错误"]),
]:
    _t = (HERE / "manuscript" / _f).read_text()
    _miss = [w for w in _must if w not in _t]
    _bad = [w for w in _mustnt if w in _t]
    claim(f"{_lang} AI 声明与作者的两次决定一致", not _miss and not _bad,
          (f"缺 {_miss}" if _miss else "") + (f" 不该有 {_bad}" if _bad else ""))

print("\nDiscussion\n")

# "they are independently rediscovered with correlations above 0.97"
k3 = pd.read_csv(HERE / "results_denovo/denovo_K3_matching.csv")
k4 = pd.read_csv(HERE / "results_denovo/denovo_K4_matching.csv")
best = {}
for tag, t in [("K=3", k3), ("K=4", k4)]:
    for s, g in t.groupby("best_match_state"):
        best[(tag, s)] = g.best_match_correlation.max()
three = [s for s in k3.best_match_state.unique()
         if "impairment-low support" not in s]
vals = {(t, s): best[(t, s)] for t in ("K=3", "K=4") for s in three if (t, s) in best}
claim("那三个状态在两个解中都被独立重发现，K=3 相关 0.971–0.991、K=4 相关 0.962–0.979",
      all(abs(min(v for (t,_),v in vals.items() if t==tag) - lo) < 0.0006 and
          abs(max(v for (t,_),v in vals.items() if t==tag) - hi) < 0.0006
          for tag, lo, hi in [("K=3", 0.971, 0.991), ("K=4", 0.962, 0.979)]),
      "; ".join(f"{t} {s[-20:]}={v:.3f}" for (t, s), v in sorted(vals.items())))

# "Two thirds of its windows are placed with the respiratory-support state under
#  both solutions -- 64.1% and 66.6% -- and only 13.0% and 20.6% with the
#  preserved state." Re-read out of the built crosstab rather than recomputed,
#  so a change in script 46 that the prose does not follow shows up here.
md = (HERE / "manuscript/denovo_crosstab.md").read_text()
for want in ["64.1", "66.6", "13.0", "20.6"]:
    claim(f"交叉表里含 {want}%", want in md)
disc = (HERE / "manuscript/discussion_jamia_EN.md").read_text()
claim("讨论引用的这四个数与交叉表一致",
      all(w in disc for w in ["64.1%", "66.6%", "13.0%", "20.6%"]))

# Cover letter: removing the organ-support variables drops only the fourth state
# below the prespecified 0.80 threshold, the other three staying above 0.97.
_tf = pd.read_csv(HERE / "results_treatment_free/per_state_transport.csv").set_index("state")
_fu = pd.read_csv(HERE / "results_full_model/per_state_full.csv")
_fu = _fu[_fu.scope == "both"].set_index("state")
_D = "neurological impairment-low support"
_others = [s_ for s_ in _tf.index if s_ != _D]
claim("去掉器官支持变量后，只有 D 跌破 0.80 门槛，另三个仍在 0.96 以上",
      _tf.loc[_D, "corr_eligible"] < 0.80
      and all(_tf.loc[s_, "corr_eligible"] > 0.96 for s_ in _others),
      f"D {_tf.loc[_D,'corr_eligible']:.3f}（完整模型 {_fu.loc[_D,'profile_corr']:.3f}）；"
      + "，".join(f"{s_[-18:]} {_tf.loc[s_,'corr_eligible']:.3f}" for s_ in _others))

# "Neither solution contained a state corresponding to neurological
#  impairment-low support, whose highest correlation with any de novo state was
#  0.183." The 0.183 must be the maximum over BOTH solutions, not just K=4.
COL = "neurological impairment-low support"
mx = max(pd.read_csv(HERE / "results_denovo" / f, index_col=0)[COL].max()
         for f in ["denovo_K3_full_corr_matrix.csv", "denovo_K4_full_corr_matrix.csv"])
claim("D 与任何 de novo 状态的最高相关为 0.183（两个解合计）",
      abs(mx - 0.183) < 0.0006, f"K=3 与 K=4 两矩阵中该列的最大值 = {mx:.3f}")

# "Fourteen of the fifteen were made with no outcome result visible."
log = pd.read_csv(HERE / "manuscript/supplementary_amendment_log.csv")
n_no = int((log.outcome_results_visible_at_the_time == "No").sum())
claim("十五条修订中十四条在无结局结果可见时做出",
      len(log) == 15 and n_no == 14, f"共 {len(log)} 条，结局不可见 {n_no} 条")

import zipfile as _zip, html as _html
_RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)

# ---- 扉页印的字数不能是陈的 --------------------------------------------------
# 加了四处图引用之后正文多了 8 个词，而扉页上的字数是手写的常数，不会跟着动。
# 一个和正文对不上的字数，编辑一眼就能看出来。
_tp = (HERE / "manuscript/title_page_EN.md").read_text(encoding="utf-8")
_stated = int(re.search(r"main text (\d+)", _tp).group(1))
_main = " ".join((HERE / "manuscript" / f).read_text(encoding="utf-8")
                 for f in ["intro_jamia_EN.md", "methods_jamia_EN.md",
                           "results_jamia_EN.md", "discussion_jamia_EN.md"])
_main = re.sub(r"^\|.*$", "", _main, flags=re.M)
_main = re.sub(r"^#+ .*$", "", _main, flags=re.M)
_counted = len(re.findall(r"\S+", re.sub(r"[*`_]", " ", _main)))
claim("扉页字数与正文实际长度相符（容差 5%）",
      abs(_stated - _counted) / _counted < 0.05,
      f"扉页 {_stated}，按同一口径数得 {_counted}")
claim("扉页字数未超 JAMIA 的 4000 词上限", _stated <= 4000, f"{_stated}/4000")
_cn = (HERE / "manuscript/title_page_CN.md").read_text(encoding="utf-8")
claim("中英扉页的字数一致", str(_stated) in _cn, f"英文 {_stated}")

# ---- 每张图和每张表都必须在正文里被引用 ------------------------------------
# JAMIA 编辑部因为这一条退回过一次投稿：四张图在正文里一次都没有被引用，
# 只出现在文末的图注里。此前 58 条检查里没有任何一条看这件事——它们都在核对
# "印出来的东西对不对"，没有人问"该被指到的东西有没有被指到"。
# 引用要在正文里数，不能算图注段，否则每张图都会"被自己引用"一次。
def _body_before_legends(path):
    x = _zip.ZipFile(path).read("word/document.xml").decode()
    t = re.sub(r"\s+", " ", _html.unescape("".join(_RUN.findall(x))))
    cut = t.rfind("Figure legends")
    if cut < 0:
        cut = t.rfind("图注")
    return t[:cut] if cut > 0 else t


for _f, _fig, _tab in [("manuscript/manuscript_JAMIA_EN.docx", "Figure", "Table"),
                       ("manuscript/manuscript_JAMIA_CN.docx", "图", "表")]:
    _b = _body_before_legends(HERE / _f)
    _miss_f = [n for n in range(1, 5)
               if not re.search(_fig + r"\s*" + str(n) + r"(?![0-9])", _b)]
    _miss_t = [n for n in range(1, 5)
               if not re.search(_tab + r"\s*" + str(n) + r"(?![0-9])", _b)]
    claim(f"{_f.split('/')[-1]} 四张图都在正文里被引用",
          not _miss_f, "全部引用" if not _miss_f else f"缺 {_miss_f}")
    claim(f"{_f.split('/')[-1]} 四张表都在正文里被引用",
          not _miss_t, "全部引用" if not _miss_t else f"缺 {_miss_t}")

# 补充表同理：S1–S5 必须在正文或图注里被指到，否则读者不知道去哪里找
_all_en = re.sub(r"\s+", " ", _html.unescape("".join(_RUN.findall(
    _zip.ZipFile(HERE / "manuscript/manuscript_JAMIA_EN.docx").read("word/document.xml").decode()))))
_miss_s = [n for n in range(1, 6)
           if not re.search(r"Supplementary Table S" + str(n) + r"(?![0-9])", _all_en)]
claim("补充表 S1–S5 都在正文里被引用", not _miss_s,
      "全部引用" if not _miss_s else f"缺 {_miss_s}")

# ---- 表格没有被分隔符拆开 -------------------------------------------------
# Read the built docx, not the markdown it came from. A cell value containing the
# pipe character is the column separator of a markdown table, and an unescaped one
# splits the cell silently: this is how every row of S1 came to be shifted, its
# stay counts, ICD codes and adjudication notes pushed out of the table, in both
# the JAMIA package and the published preprint. Counting cells in the generator's
# own output would not have caught it — the generator is what produced the split.
def _tables(path):
    x = _zip.ZipFile(path).read("word/document.xml").decode()
    for t in re.findall(r"<w:tbl>.*?</w:tbl>", x, re.S):
        yield [[_html.unescape("".join(_RUN.findall(c)))
                for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S)]
               for r in re.findall(r"<w:tr[ >].*?</w:tr>", t, re.S)]


for _f in ["manuscript/supplementary_tables.docx",
           "manuscript/manuscript_JAMIA_EN.docx",
           "manuscript/manuscript_JAMIA_CN.docx"]:
    _ragged = [(i, j, len(r), len(t[0]))
               for i, t in enumerate(_tables(HERE / _f))
               for j, r in enumerate(t) if len(r) != len(t[0])]
    claim(f"{_f.split('/')[-1]} 每张表每一行的单元格数都与表头相同",
          not _ragged, "无参差" if not _ragged else f"{len(_ragged)} 行不符: {_ragged[:3]}")

# S1 must carry its adjudication, not just the path: 35 paths, 6 columns, and a
# stay count on every row that agrees with the phenotype table it was built from.
_s1 = next(t for t in _tables(HERE / "manuscript/supplementary_tables.docx")
           if t[0][0].startswith("diagnosisstring"))
_body = _s1[1:]
_src = pd.read_csv(HERE / "frozen_phenotype/phenotype_paths_frozen.csv")
_ok = (len(_body) == 35 and all(len(r) == 6 for r in _body)
       and all(r[3].isdigit() for r in _body)
       and all(r[0].count("|") >= 2 for r in _body))
claim("S1 完整：35 条路径、6 列、每行都有整数 Stays、路径中的竖线未被当作分隔符",
      _ok, f"{len(_body)} 行；Stays 合计 {sum(int(r[3]) for r in _body if r[3].isdigit())}")

claim("S1 的 Stays 与裁定表逐行一致",
      sorted(int(r[3]) for r in _body) == sorted(int(v) for v in _src.n_stays),
      f"docx 合计 {sum(int(r[3]) for r in _body)}，来源合计 {int(_src.n_stays.sum())}")

print()
bad = [t for ok, t, _ in R if not ok]
print(f"{len(R) - len(bad)}/{len(R)} 条论断通过")
if bad:
    print("\n❌ 站不住的论断：")
    for t in bad:
        print("   ", t)
    raise SystemExit(1)
