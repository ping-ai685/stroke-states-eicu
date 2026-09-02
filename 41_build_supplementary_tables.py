"""
Paper 2: build the four supplementary tables the manuscript cites.

All four were referenced in the text before any of them existed as a file, which
is the same defect in four places. Each is generated here from the frozen
pipeline output it describes, so none can drift from what the analysis actually
did:

  S1  the 35 enumerated diagnosisstring paths of the frozen stroke phenotype
  S2  the 21-variable harmonization mapping, with measured window availability
  S3  the amendment log, fifteen entries with what was visible at the time
  S4  the paired prediction contrasts, every scope and every horizon

Output: manuscript/supplementary_tables.md
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "manuscript"
PRED = HERE / "prediction"


def md(df, cols=None, headers=None):
    d = df[cols] if cols else df
    hdr = headers or list(d.columns)
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


blocks = []

# ---- S1 -----------------------------------------------------------------------
s1 = pd.read_csv(HERE / "frozen_phenotype/phenotype_paths_frozen.csv")
s1 = s1.sort_values(["status", "subtypes", "diagnosisstring"])
blocks.append((
    "**Supplementary Table S1.** The frozen stroke phenotype: every `diagnosisstring` "
    "path in eICU-CRD reviewed for this study, with the subtype it was assigned, the "
    "include or exclude decision, and the number of stays it contributed. The list was "
    "enumerated, clinically adjudicated and frozen before any state was assigned. ICD "
    "codes are shown for reference only and were not used for case definition: in eICU "
    "`icd9code` is a deterministic re-encoding of `diagnosisstring`, so it carries no "
    "independent information (amendment A).",
    md(s1, ["diagnosisstring", "subtypes", "status", "n_stays", "icd_codes_attached", "note"],
       ["diagnosisstring path", "Subtype", "Decision", "Stays", "ICD codes attached (reference)", "Note"])))

# ---- S2 -----------------------------------------------------------------------
s2 = pd.read_csv(HERE / "harmonization_dictionary.csv")
av = pd.read_csv(HERE / "audit/availability_vs_mimic.csv").set_index("variable")
s2 = s2.assign(
    mimic_pct=s2.variable.map(av.mimic_pct_windows),
    eicu_pct=s2.variable.map(av.eicu_pct_windows),
    diff=s2.variable.map(av.difference))
for c in ("mimic_pct", "eicu_pct", "diff"):
    s2[c] = s2[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
blocks.append((
    "**Supplementary Table S2.** Harmonization of the 21 model variables from MIMIC-IV to "
    "eICU-CRD: source table, field, selector, unit handling, offset field and window "
    "aggregation rule for each, with the measured window-level availability in both "
    "databases computed identically and before any forward fill. Tier A variables map "
    "directly; tier B require validation; tier C require algorithmic reconstruction from "
    "free text and are covered by the frozen term dictionary. Availability is blank for the "
    "four organ-support variables, which are constructed rather than measured.",
    md(s2, ["tier", "variable", "eicu_table", "eicu_field", "eicu_selector",
            "unit_transformation", "window_aggregation", "mimic_pct", "eicu_pct", "diff"],
       ["Tier", "Variable", "eICU table", "Field", "Selector", "Unit handling",
        "Window aggregation", "MIMIC-IV, %", "eICU, %", "Difference"])))

# ---- S3 -----------------------------------------------------------------------
s3 = pd.read_csv(OUT / "supplementary_amendment_log.csv")
blocks.append((
    "**Supplementary Table S3.** The amendment log. Every change to the protocol made "
    "during extraction and analysis, with the date, the section affected, what triggered "
    "it, the decision taken, and — the two columns that matter for interpretation — whether "
    "state assignments and outcome results were visible at the time it was made. Fourteen "
    "of the fifteen were made with no outcome result visible in either database. The "
    "fifteenth, amendment O, added Aim 4 after Aims 1 to 3 were complete and is the only "
    "entry for which both columns read yes; it is reported as exploratory throughout.",
    md(s3, ["amendment", "date", "section", "trigger",
            "state_assignments_visible_at_the_time", "outcome_results_visible_at_the_time",
            "decision", "classification"],
       ["#", "Date", "Section", "Trigger", "States visible?", "Outcomes visible?",
        "Decision", "Classification"])))

# ---- S4 -----------------------------------------------------------------------
LAB = {"L1 - L0": "L1 − L0  frozen transition structure over persistence",
       "L2 - L1": "L2 − L1  observation history over the decoded label",
       "L3 - L2": "L3 − L2  fitting a predictor over the frozen model",
       "L4 - L3": "L4 − L3  relaxing linearity",
       "L5 - L4": "L5 − L4  adding baseline covariates",
       "L5 - L2": "L5 − L2  everything gained by fitting"}
SCOPE = {"mimic-test": "MIMIC-IV test", "eicu-full": "eICU, full",
         "eicu-gcs-eligible": "eICU, GCS-eligible",
         "eicu-vent-ascertainable": "eICU, vent-ascertainable",
         "eicu-both": "eICU, both rules"}
c = pd.read_csv(PRED / "prediction_contrasts.csv")
c["cell"] = c.apply(lambda r: f"{r.delta_auroc:+.4f} ({r.lo:+.4f}, {r.hi:+.4f})", axis=1)
piv = c.pivot(index="contrast", columns="scope", values="cell").loc[list(LAB)]
piv.index = [LAB[i] for i in piv.index]
piv = piv[list(SCOPE)]
piv.columns = [SCOPE[x] for x in piv.columns]
a = md(piv.reset_index(), headers=["Contrast at a 6-hour horizon"] + list(piv.columns))

mh = pd.read_csv(PRED / "multihorizon_metrics.csv")
MN = {"L0": "Persistence (null)", "L1": "Frozen transition matrix",
      "L2": "Frozen HMM, filtered", "L3": "Multinomial logistic",
      "L4": "Gradient boosting", "L5": "Gradient boosting + covariates",
      "L6": "Sequence model (GRU)"}
l6 = pd.read_csv(PRED / "metrics_L6.csv")[["scope", "horizon_h", "change_auroc"]].assign(model="L6")
mh = pd.concat([mh[mh.model.isin(MN)][["scope", "horizon_h", "model", "change_auroc"]], l6],
               ignore_index=True)
mh["cell"] = mh.change_auroc.map(lambda v: f"{v:.3f}")
# Scope order follows the rest of the paper, not the alphabet; L6 was not run in
# the both-rules scope, so those cells are left blank rather than imputed.
ORD = [(sc, h) for sc in ["mimic-test", "eicu-full", "eicu-both"] for h in [6, 12, 24]]
p2 = mh.pivot_table(index="model", columns=["scope", "horizon_h"], values="cell", aggfunc="first")
p2 = p2.reindex(index=list(MN), columns=pd.MultiIndex.from_tuples(ORD))
p2.index = [MN[i] for i in p2.index]
hdr = ["Predictor"] + [f"{SCOPE[s]}, {h} h" for s, h in p2.columns]
b = md(p2.reset_index(), headers=hdr)

blocks.append((
    "**Supplementary Table S4.** Prediction results across every analysis scope. "
    "*Upper panel:* paired differences between adjacent rungs of the ladder at a six-hour "
    "horizon, each recomputed within the same patient-clustered resample. Every difference "
    "excludes zero in both databases and in all four eICU scopes, which is the claim made "
    "in Table 8. *Lower panel:* area under the ROC curve for the event that the state "
    "changes, by predictor, scope and horizon. The horizon analysis was run in the two "
    "primary scopes and the strictest one, and the sequence model in the two primary "
    "scopes only; the ordering of predictors is identical throughout.",
    a + "\n\n" + b))

(OUT / "supplementary_tables.md").write_text(
    "# Supplementary tables\n\n" + "\n\n\n".join(f"{c_}\n\n{t}" for c_, t in blocks) + "\n")
print(f"wrote {OUT/'supplementary_tables.md'}")
for i, (c_, t) in enumerate(blocks, 1):
    print(f"  S{i}: {len([l for l in t.splitlines() if l.startswith('|')]) - 2} data rows")
