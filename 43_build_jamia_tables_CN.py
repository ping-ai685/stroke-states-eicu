"""
Paper 2: JAMIA 投稿版的四张中文正文表。

与脚本 42 同构：表 1、2、3 直接取自完整版中文表格（队列、迁移标准、结局关联），
仅重新编号，不重新排版，故两版不可能在数值上分叉；表 4 为预测结果，含两个面板。

输出: manuscript/tables_jamia_CN.md
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "manuscript"
PRED = HERE / "prediction"


def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] +
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


full = (OUT / "tables_v1_CN.md").read_text()


def lift(n):
    i = full.index(f"**表 {n}.**")
    j = full.find("**表 ", i + 6)
    return full[i:(j if j != -1 else len(full))].strip()


def renumber(block, new):
    old = block[len("**表 "):block.index(".")]
    return block.replace(f"**表 {old}.**", f"**表 {new}.**", 1)


blocks = [renumber(lift(1), 1), renumber(lift(3), 2), renumber(lift(5), 3)]

# ---- 表 4：预测，两个面板 -----------------------------------------------------
M = pd.read_csv(PRED / "metrics_by_model.csv")
L6 = pd.read_csv(PRED / "metrics_L6.csv")
MH = pd.read_csv(PRED / "multihorizon_metrics.csv")
SC = pd.read_csv(PRED / "sequence_contrasts.csv")
mt = M[M.scope == "mimic-test"].set_index("model")
ef = M[M.scope == "eicu-full"].set_index("model")
CN = {"L0": "持续（零模型）", "L1": "冻结转换矩阵", "L2": "冻结 HMM，前向滤波",
      "L3": "多项逻辑回归", "L4": "梯度提升", "L5": "梯度提升 + 协变量"}
INFO = {"L0": "仅当前状态，假定延续", "L1": "解码出的当前状态",
        "L2": "截至 *t* 的完整观测历史", "L3": "当前状态 + *t* 时刻 21 个变量",
        "L4": "同 L3，不设线性假定", "L5": "同 L4 + 年龄、性别、亚型、窗口序号"}
FIT = {"L0": "否", "L1": "否", "L2": "否", "L3": "是", "L4": "是", "L5": "是"}


def ci(d, c="change_auroc"):
    return f"{d[c]:.3f}（{d[f'{c}_lo']:.3f}–{d[f'{c}_hi']:.3f}）"


rows = [[CN[k], INFO[k], FIT[k], ci(mt.loc[k]), ci(ef.loc[k]),
         f"{ef.loc[k,'change_sens_at_90spec']*100:.1f}"] for k in CN]
l6m = L6[(L6.scope == "mimic-test") & (L6.horizon_h == 6)].iloc[0]
l6e = L6[(L6.scope == "eicu-full") & (L6.horizon_h == 6)].iloc[0]
rows.append(["序列模型（GRU）", "截至 *t* 的原始观测序列", "是",
             ci(l6m), ci(l6e), f"{l6e.change_sens_at_90spec*100:.1f}"])
pa = md(rows, ["预测器", "所用信息", "是否拟合", "MIMIC-IV 测试集 AUROC（95% CI）",
               "eICU AUROC（95% CI）", "eICU 90% 特异度下灵敏度 %"])

rows = []
for h in [6, 12, 24]:
    e = MH[(MH.scope == "eicu-full") & (MH.horizon_h == h)].set_index("model")
    q = L6[(L6.scope == "eicu-full") & (L6.horizon_h == h)].iloc[0]
    sc = SC[(SC.scope == "eicu-full") & (SC.horizon_h == h)].iloc[0]
    share = (e.loc["L2", "change_auroc"] - 0.5) / (q.change_auroc - 0.5) * 100
    rows.append([f"{h} 小时", f"{int(e.loc['L2','pairs']):,}", f"{e.loc['L2','changed_pct']:.1f}",
                 f"{e.loc['L2','change_auroc']:.3f}", f"{q.change_auroc:.3f}",
                 f"+{sc.delta_auroc:.3f}（{sc.lo:+.3f} 至 {sc.hi:+.3f}）", f"{share:.1f}"])
pb = md(rows, ["跨度", "配对数", "发生变化 %", "冻结 HMM", "序列模型",
               "序列模型 − 冻结（95% CI）", "保留的判别力 %"])

cap = ("**表 4.** 下一状态的预测。所预测的事件为「下一窗口的状态与当前不同」，在 eICU 中"
       "发生于 9.3% 的配对；因此一条只预测「不变」的规则准确率达 0.906，却标记不出任何人，"
       "故改以该事件的判别力衡量。*面板 A，6 小时跨度：* 预测器按各自被允许使用的信息排序。"
       "前三者不涉及任何拟合，是发现模型被用于另一种操作；其余在 MIMIC-IV 训练集上拟合，"
       "并在解码 eICU 之前冻结。纳入序列模型是为了划定天花板，其为三个种子的平均。"
       "*面板 B，eICU 各跨度：* 冻结转换矩阵取相应次幂；拟合类预测器按每个跨度在 MIMIC-IV 训练集上"
       "单独训练、并在外部评估之前冻结，从未在 eICU 中重新拟合。判别力上升而非衰减，而「保留的判别力」"
       "——完全不做拟合即达到的、随机水平之上的判别力，占序列模型相应值的百分比；这是判别力的份额，而非信息量的份额——亦随之上升，"
       "说明四状态所丢弃的信息集中在短时程。区间为按患者聚类的自助百分位；跨度对比在同一次"
       "重抽样内配对完成。全部分析范围的完整结果见 Supplementary Table S4。")

blocks.append(f"{cap}\n\n{pa}\n\n{pb}")
(OUT / "tables_jamia_CN.md").write_text("# 表\n\n" + "\n\n\n".join(blocks) + "\n")

# 行数必须与英文版逐表一致
en = (OUT / "tables_jamia.md").read_text()
def nrows(b):
    return len([l for l in b.splitlines() if l.startswith("|")])
en_blocks = [c[c.find("**Table "):].strip() for c in en.split("\n\n\n") if "**Table " in c]
for i, (b, e) in enumerate(zip(blocks, en_blocks), 1):
    assert nrows(b) == nrows(e), f"表 {i} 行数不一致：中文 {nrows(b)}，英文 {nrows(e)}"
print(f"wrote {OUT/'tables_jamia_CN.md'} — 四张表行数与英文版一致")
