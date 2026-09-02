"""
Display text for the manuscript figures, in both languages.

The figures are drawn once and labelled twice. Keeping the strings here rather
than inline means the English and Chinese sets are produced by the same plotting
code, so they cannot diverge the way two hand-maintained versions would.

Usage:  LANG = "EN" | "CN";  T = labels(LANG);  T["fig2_xlabel"]

Note on the Chinese set: Hiragino Sans GB has no glyph for U+2212, so Chinese
labels use the ASCII hyphen. `axes.unicode_minus = False` handles tick labels.
"""

FONT = {"EN": "DejaVu Sans", "CN": "Hiragino Sans GB"}

_EN = {
    # states
    "st_P": "Preserved–\nlow support",
    "st_L": "Impairment–\nlow support",
    "st_N": "Impairment–\nrenal dysfunction",
    "st_R": "Impairment–\nrespiratory support",
    # figure 1
    "f1_source": "eICU-CRD v2.0\n200,859 ICU unit stays · 208 hospitals",
    "f1_pheno": "Frozen stroke phenotype\n35 enumerated diagnosis paths\n{n:,} stays · {h} hospitals",
    "f1_cohort": "Analysis cohort\n8,279 patients · 176 hospitals\n70,630 six-hour windows",
    "f1_gcs": "GCS-eligible\n≥20 patients and GCS\nin ≥50% of windows\n39 hospitals\n4,568 patients",
    "f1_vent": "Ventilation-ascertainable\nboth determining\ninterfaces ≥10% of stays\n140 hospitals\n7,500 patients",
    "f1_both": "Both rules\nstrictest scope, and the\nbetween-hospital denominator\n38 hospitals\n4,296 patients",
    "f1_rules": "Hospital eligibility rules, frozen before any state assignment",
    "f1_stays": "{n:,} stays",
    "f1_x_trauma": "Excluded {n:,}\nhead, face or CNS trauma",
    "f1_x_age": "Excluded {n:,}\nage <18 or age unparseable",
    "f1_x_first": "Excluded {n:,}\nnot the first ICU unit stay",
    "f1_x_12h": "Excluded {n:,}\nICU stay <12 h",
    "f1_x_second": "Excluded {n:,}\nsecond qualifying hospitalization",
    # figure 2
    "f2_xlabel": "MIMIC-IV state mean",
    "f2_ylabel": "eICU state mean",
    "f2_cont": "continuous (z)",
    "f2_org": "organ support (proportion)",
    "f2_mimic": "MIMIC-IV (discovery)",
    "f2_eicu": "eICU (both eligibility rules)",
    "f2_pct": "% of six-hour windows",
    # figure 3
    "f3_cbar": "Profile correlation with discovery state",
    "f3_k3": "eICU de novo, K = 3\n(most stable: restart ARI 0.991)",
    "f3_k4": "eICU de novo, K = 4\n(restart ARI 0.840)",
    "f3_state": "eICU state ",
    "f3_note": ("Outlined cell: each eICU state's closest discovery state. Given three states, eICU finds preserved, renal\n"
                "dysfunction and respiratory support. Given four, it subdivides the preserved state rather than recovering a fourth.\n"
                "Dashed box: both eICU states match the same discovery state. Neither solution contains a counterpart\n"
                "to neurological impairment–low support."),
    # figure 4
    "f4_A": "A   The ladder at a 6-hour horizon",
    "f4_B": "B   The gap closes with the horizon (eICU)",
    "f4_C": "C   Enrichment over chance is about twofold",
    "f4_xlabel": "AUROC for “the state changes in the next window”",
    "f4_auroc": "AUROC",
    "f4_ap": "Average precision",
    "f4_nodisc": "no discrimination",
    "f4_nofit": "no fitting",
    "f4_share": "teal labels: frozen HMM as a share of the sequence model",
    "f4_mimic": "MIMIC-IV test split",
    "f4_eicu": "eICU (176 hospitals)",
    "f4_seq": "Sequence model",
    "f4_win": "Best single-window",
    "f4_frozen": "Frozen HMM",
    "f4_base": "Base rate (chance)",
    "L0": "Persistence (null)", "L1": "Frozen transition matrix",
    "L2": "Frozen HMM, filtered", "L3": "Multinomial logistic",
    "L4": "Gradient boosting", "L5": "Gradient boosting + covariates",
    "L6": "Sequence model (GRU)",
    "hr": " h",
}

_CN = {
    "st_P": "保留–\n低支持",
    "st_L": "受损–\n低支持",
    "st_N": "受损–\n肾功能障碍",
    "st_R": "受损–\n呼吸支持",
    "f1_source": "eICU-CRD v2.0\n200,859 次 ICU 入科 · 208 家医院",
    "f1_pheno": "冻结的卒中表型\n35 条枚举诊断路径\n{n:,} 次入科 · {h} 家医院",
    "f1_cohort": "分析队列\n8,279 例患者 · 176 家医院\n70,630 个 6 小时窗口",
    "f1_gcs": "GCS 合格\n≥20 例患者且 GCS\n覆盖 ≥50% 窗口\n39 家医院\n4,568 例患者",
    "f1_vent": "通气可判定\n两个判定接口\n均覆盖 ≥10% 入科\n140 家医院\n7,500 例患者",
    "f1_both": "同时满足两条\n最严格范围，亦为\n院间分析的分母\n38 家医院\n4,296 例患者",
    "f1_rules": "医院准入规则，在任何状态分配之前冻结",
    "f1_stays": "{n:,} 次入科",
    "f1_x_trauma": "排除 {n:,}\n头面部或中枢神经系统创伤",
    "f1_x_age": "排除 {n:,}\n年龄 <18 或年龄无法解析",
    "f1_x_first": "排除 {n:,}\n非首次 ICU 入科",
    "f1_x_12h": "排除 {n:,}\nICU 停留 <12 小时",
    "f1_x_second": "排除 {n:,}\n第二次符合条件的住院",
    "f2_xlabel": "MIMIC-IV 状态均值",
    "f2_ylabel": "eICU 状态均值",
    "f2_cont": "连续变量（z）",
    "f2_org": "器官支持（比例）",
    "f2_mimic": "MIMIC-IV（发现队列）",
    "f2_eicu": "eICU（同时满足两条规则）",
    "f2_pct": "占 6 小时窗口的百分比",
    "f3_cbar": "与发现状态的特征谱相关系数",
    "f3_k3": "eICU 独立重拟合，K = 3\n（最稳定：重启 ARI 0.991）",
    "f3_k4": "eICU 独立重拟合，K = 4\n（重启 ARI 0.840）",
    "f3_state": "eICU 状态 ",
    "f3_note": ("描边单元格：各 eICU 状态最接近的发现状态。给定三个状态时，eICU 找出保留态、肾功能障碍态\n"
                "与呼吸支持态；给定四个时，它细分保留态，而非找回第四个状态。\n"
                "虚线框：两个 eICU 状态匹配到同一个发现状态。两个解中都不存在与\n"
                "神经功能受损–低支持相对应的状态。"),
    "f4_A": "A   6 小时跨度上的信息阶梯",
    "f4_B": "B   差距随跨度收窄（eICU）",
    "f4_C": "C   相对随机的富集约为两倍",
    "f4_xlabel": "对「下一窗口状态改变」的 AUROC",
    "f4_auroc": "AUROC",
    "f4_ap": "平均精度",
    "f4_nodisc": "无判别力",
    "f4_nofit": "不做拟合",
    "f4_share": "青色标签：冻结 HMM 占序列模型的份额",
    "f4_mimic": "MIMIC-IV 测试集",
    "f4_eicu": "eICU（176 家医院）",
    "f4_seq": "序列模型",
    "f4_win": "最优单窗口模型",
    "f4_frozen": "冻结 HMM",
    "f4_base": "基线发生率（随机）",
    "L0": "持续（零模型）", "L1": "冻结转换矩阵",
    "L2": "冻结 HMM，前向滤波", "L3": "多项逻辑回归",
    "L4": "梯度提升", "L5": "梯度提升 + 协变量",
    "L6": "序列模型（GRU）",
    "hr": " 小时",
}

assert set(_EN) == set(_CN), sorted(set(_EN) ^ set(_CN))


def labels(lang):
    assert lang in ("EN", "CN"), lang
    return dict(_EN if lang == "EN" else _CN)
