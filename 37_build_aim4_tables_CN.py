"""
Paper 2, Aim 4: 表 7 至表 9 的中文版。

与脚本 33 读同一批 CSV，行数与英文版逐表核对；核对失败即中止，避免两个语种
的表在数值上悄悄分叉。

输出: manuscript/tables_aim4_CN.md
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
PRED = HERE / "prediction"
OUT = HERE / "manuscript"

M = pd.read_csv(PRED / "metrics_by_model.csv")
C = pd.read_csv(PRED / "prediction_contrasts.csv")
QA = pd.read_csv(PRED / "dataset_qa.csv").set_index("scope")
MH = pd.read_csv(PRED / "multihorizon_metrics.csv")
HC = pd.read_csv(PRED / "horizon_contrasts.csv")
FX = pd.read_csv(PRED / "horizon_sensitivity.csv")
fx = FX[FX.analysis == "population fixed at 24 h"]

CN = {"L0": "L0 持续（零模型）", "L1": "L1 冻结转换矩阵", "L2": "L2 冻结 HMM，前向滤波",
      "L3": "L3 多项逻辑回归", "L4": "L4 梯度提升", "L5": "L5 梯度提升 + 协变量"}
INFO = {"L0": "仅当前状态，假定延续", "L1": "解码出的当前状态",
        "L2": "截至 *t* 的完整观测历史", "L3": "当前状态 + *t* 时刻 21 个变量",
        "L4": "同 L3，不设线性假定", "L5": "同 L4 + 年龄、性别、亚型、窗口序号"}
FIT = {"L0": "否", "L1": "否", "L2": "否", "L3": "是", "L4": "是", "L5": "是"}
ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]


def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] +
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def ci(d, col):
    return f"{d[col]:.3f}（{d[f'{col}_lo']:.3f}–{d[f'{col}_hi']:.3f}）"


mt = M[M.scope == "mimic-test"].set_index("model")
ef = M[M.scope == "eicu-full"].set_index("model")
L6 = pd.read_csv(PRED / "metrics_L6.csv")
l6m = L6[(L6.scope == "mimic-test") & (L6.horizon_h == 6)].iloc[0]
l6e = L6[(L6.scope == "eicu-full") & (L6.horizon_h == 6)].iloc[0]
rows7 = [[CN[k], INFO[k], FIT[k], ci(mt.loc[k], "change_auroc"), ci(ef.loc[k], "change_auroc"),
          f"{ef.loc[k,'change_auprc']:.3f}", f"{ef.loc[k,'change_sens_at_90spec']*100:.1f}",
          f"{ef.loc[k,'accuracy']:.3f}"] for k in ORDER]
rows7.append(["L6 序列模型（GRU）", "截至 *t* 的原始观测序列，及协变量", "是",
              ci(l6m, "change_auroc"), ci(l6e, "change_auroc"),
              f"{l6e.change_auprc:.3f}", f"{l6e.change_sens_at_90spec*100:.1f}",
              f"{l6e.accuracy:.3f}"])
T7 = ("**表 7.** 按信息集划分的下一状态一步预测。所预测的事件为"
      "「下一个 6 小时窗口的状态与当前不同」，在 MIMIC-IV 测试集中发生率为 "
      f"{QA.loc['mimic-test','changed_pct']}%，在 eICU 中为 {QA.loc['eicu-full','changed_pct']}%。"
      "持续（零模型）对每个窗口给出相同概率，故按构造对该事件无判别力；"
      "总体准确率因此列在最后而非最前——它在零模型与最优模型之间几无差别。"
      "L0 至 L2 为冻结的发现模型用于另一种操作，不涉及任何拟合；"
      "L3 至 L6 仅在 MIMIC-IV 训练集上拟合，并在解码 eICU 之前冻结。"
      "纳入 L6 是为了用一个直接读取观测序列、而非逐窗口读取的模型来划定天花板；"
      "其取三个种子的平均，种子间极差为 0.009。区间为按患者聚类的自助百分位。",
      md(rows7, ["预测器", "所用信息", "是否拟合", "MIMIC-IV 测试集 AUROC（95% CI）",
                 "eICU AUROC（95% CI）", "eICU AUPRC", "eICU 90% 特异度下灵敏度 %", "eICU 准确率"]))

LAB = {"L1 - L0": "L1 − L0：冻结转换结构相对持续", "L2 - L1": "L2 − L1：观测历史相对解码标签",
       "L3 - L2": "L3 − L2：拟合预测器相对冻结模型", "L4 - L3": "L4 − L3：放宽线性假定",
       "L5 - L4": "L5 − L4：加入基线协变量", "L5 - L2": "L5 − L2：拟合带来的全部增益"}
cm = C[C.scope == "mimic-test"].set_index("contrast")
ce = C[C.scope == "eicu-full"].set_index("contrast")
rows8 = [[LAB[k], f"{cm.loc[k,'delta_auroc']:+.4f}（{cm.loc[k,'lo']:+.4f} 至 {cm.loc[k,'hi']:+.4f}）",
          f"{ce.loc[k,'delta_auroc']:+.4f}（{ce.loc[k,'lo']:+.4f} 至 {ce.loc[k,'hi']:+.4f}）"] for k in LAB]
T8 = ("**表 8.** 预测梯子相邻层级之间的配对差值。各模型在完全相同的窗口配对上评估，"
      "故每个差值均在同一次患者聚类重抽样内重新计算；所得区间为配对区间，"
      "较表 7 中两个边际区间的比较更窄。每一个差值在两个数据库、"
      "四个 eICU 分析范围内均排除零（Supplementary Table S4）。最大的单项增益出现在第一步："
      "冻结的转换结构，所依据的仅是解码出的当前状态。",
      md(rows8, ["对比", "MIMIC-IV 测试集 Δ AUROC（95% CI）", "eICU Δ AUROC（95% CI）"]))

rows9 = []
SEQ = pd.read_csv(PRED / "metrics_L6.csv")
SC = pd.read_csv(PRED / "sequence_contrasts.csv")
for h in [6, 12, 24]:
    e = MH[(MH.scope == "eicu-full") & (MH.horizon_h == h)].set_index("model")
    q = SEQ[(SEQ.scope == "eicu-full") & (SEQ.horizon_h == h)].iloc[0]
    sc = SC[(SC.scope == "eicu-full") & (SC.horizon_h == h)].iloc[0]
    win = e.loc[["L3", "L4", "L5"], "change_auroc"].max()
    share = (e.loc["L2", "change_auroc"] - 0.5) / (q.change_auroc - 0.5) * 100
    f = fx[(fx.scope == "eicu-full") & (fx.horizon_h == h)].iloc[0]
    rows9.append([f"{h} 小时", f"{int(e.loc['L2','pairs']):,}", f"{e.loc['L2','changed_pct']:.1f}",
                  f"{e.loc['L2','change_auroc']:.3f}", f"{win:.3f}", f"{q.change_auroc:.3f}",
                  f"+{sc.delta_auroc:.3f}（{sc.lo:+.3f} 至 {sc.hi:+.3f}）",
                  f"{share:.1f}", f"{f['L2']:.3f}"])

T9 = ("**表 9.** 判别力随预测跨度的变化，eICU。冻结转换矩阵取相应次幂；对不拟合的预测器而言"
      "此外别无改动，拟合类预测器则按每个跨度在 MIMIC-IV 训练集上重新拟合。"
      "判别力并未随跨度衰减而是上升；与之同步变动的有两个量，必须一并读取："
      "事件变得更常见，且可用集合收缩至住院时长足以到达该跨度的病例。"
      "末列在人群固定于 24 小时可用集合的条件下重复冻结模型——三个跨度使用同一批患者与同一批索引窗口，"
      "仅目标在移动；上升依然存在且略陡，故并非由该收缩造成。"
      "天花板由序列模型划定，它直接读取观测历史；单窗口模型并列于旁，因为它们够不到该天花板。"
      "「冻结份额」指完全不做拟合即达到的、零模型之上的判别力，占序列模型所达水平的百分比；"
      "该份额随跨度上升，说明四状态所丢弃的信息集中在短时程。",
      md(rows9, ["跨度", "配对数", "发生变化 %", "冻结 HMM 滤波", "最优单窗口模型", "序列模型",
                 "序列模型 − 冻结 HMM（95% CI）", "冻结份额 %", "冻结 HMM，人群固定"]))

# 行数必须与英文版一致
en = (OUT / "tables_aim4.md").read_text()
def datarows(block):
    return len([ln for ln in block.splitlines()
                if ln.startswith("|") and not set(ln) <= set("|- ")]) - 1
for n, T in [("7", T7), ("8", T8), ("9", T9)]:
    n_cn = datarows(T[1])
    n_en = datarows(en.split(f"**Table {n}.**")[1].split("**Table ")[0])
    assert n_cn == n_en, f"表 {n} 数据行数不一致：中文 {n_cn}，英文 {n_en}"

(OUT / "tables_aim4_CN.md").write_text(
    "# Aim 4 表\n\n" + f"{T7[0]}\n\n{T7[1]}\n\n\n{T8[0]}\n\n{T8[1]}\n\n\n{T9[0]}\n\n{T9[1]}\n")
print(f"wrote {OUT/'tables_aim4_CN.md'} — 表 7、8、9 行数与英文版一致")
