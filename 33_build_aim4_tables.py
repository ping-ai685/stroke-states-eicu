"""
Paper 2, Aim 4: build Tables 7 and 8 as markdown, for insertion into the prose.

Same discipline as step 25: the tables are written INTO the manuscript sources
so that the per-table submission files can be extracted from the built document
by step 28 rather than rendered separately from the CSVs.

Output: manuscript/tables_aim4.md
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
PRED = HERE / "prediction"
OUT = HERE / "manuscript"

M = pd.read_csv(PRED / "metrics_by_model.csv")
C = pd.read_csv(PRED / "prediction_contrasts.csv")
QA = pd.read_csv(PRED / "dataset_qa.csv").set_index("scope")

INFO = {"L0": "current state, assumed to continue",
        "L1": "decoded current state",
        "L2": "full observation history through *t*",
        "L3": "current state + 21 variables at *t*",
        "L4": "as L3, without a linearity assumption",
        "L5": "as L4 + age, sex, subtype, window index"}
FIT = {"L0": "no", "L1": "no", "L2": "no", "L3": "yes", "L4": "yes", "L5": "yes"}
NAME = dict(zip(M.model, M.model_name))
ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]


def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] +
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def ci(d, col):
    return f"{d[col]:.3f} ({d[f'{col}_lo']:.3f}–{d[f'{col}_hi']:.3f})"


mt = M[M.scope == "mimic-test"].set_index("model")
ef = M[M.scope == "eicu-full"].set_index("model")
L6 = pd.read_csv(PRED / "metrics_L6.csv")
l6m = L6[(L6.scope == "mimic-test") & (L6.horizon_h == 6)].iloc[0]
l6e = L6[(L6.scope == "eicu-full") & (L6.horizon_h == 6)].iloc[0]

rows = [[f"{k}. {NAME[k]}", INFO[k], FIT[k], ci(mt.loc[k], "change_auroc"),
         ci(ef.loc[k], "change_auroc"), f"{ef.loc[k,'change_auprc']:.3f}",
         f"{ef.loc[k,'change_sens_at_90spec']*100:.1f}",
         f"{ef.loc[k,'accuracy']:.3f}"] for k in ORDER]
rows.append(["L6. sequence model (GRU)", "the raw observation sequence through *t*, and covariates",
             "yes", ci(l6m, "change_auroc"), ci(l6e, "change_auroc"),
             f"{l6e.change_auprc:.3f}", f"{l6e.change_sens_at_90spec*100:.1f}",
             f"{l6e.accuracy:.3f}"])

T7 = ("**Table 7.** One-step-ahead prediction of the next state, by information set. "
      "The event predicted is that the state differs in the following 6-hour window; it "
      f"occurs in {QA.loc['mimic-test','changed_pct']}% of MIMIC-IV test pairs and "
      f"{QA.loc['eicu-full','changed_pct']}% of eICU pairs. Persistence assigns the same "
      "probability to every window and is therefore uninformative for this event by "
      "construction, which is why overall accuracy — where it equals the best model — is "
      "reported last rather than first. L0 to L2 are the frozen discovery model used for a "
      "different operation and involve no fitting; L3 to L6 are fitted on the MIMIC-IV "
      "training split alone and frozen before eICU is decoded. L6 is included to set the "
      "ceiling with a model that reads the observation sequence directly rather than one "
      "window at a time; it is the mean of three seeds, whose spread was 0.005. Intervals "
      "are patient-clustered bootstrap percentiles.",
      md(rows, ["Predictor", "Information used", "Fitted", "MIMIC-IV test, AUROC (95% CI)",
                "eICU, AUROC (95% CI)", "eICU AUPRC", "eICU sens. at 90% spec., %",
                "eICU accuracy"]))

LAB = {"L1 - L0": "L1 − L0: frozen transition structure over persistence",
       "L2 - L1": "L2 − L1: observation history over the decoded label",
       "L3 - L2": "L3 − L2: fitting a predictor over the frozen model",
       "L4 - L3": "L4 − L3: relaxing linearity",
       "L5 - L4": "L5 − L4: adding baseline covariates",
       "L5 - L2": "L5 − L2: everything gained by fitting"}
cm = C[C.scope == "mimic-test"].set_index("contrast")
ce = C[C.scope == "eicu-full"].set_index("contrast")
rows = [[LAB[k], f"{cm.loc[k,'delta_auroc']:+.4f} ({cm.loc[k,'lo']:+.4f} to {cm.loc[k,'hi']:+.4f})",
         f"{ce.loc[k,'delta_auroc']:+.4f} ({ce.loc[k,'lo']:+.4f} to {ce.loc[k,'hi']:+.4f})"]
        for k in LAB]

T8 = ("**Table 8.** Paired differences between adjacent rungs of the prediction ladder. "
      "The models are evaluated on identical window pairs, so each difference is "
      "recomputed within the same patient-clustered resample; the interval is paired and "
      "is narrower than a comparison of the two marginal intervals in Table 7 would "
      "suggest. Every difference excludes zero in both databases and in all four eICU "
      "analysis scopes (Supplementary Table S4). The single largest gain is the first: "
      "the frozen transition structure, applied to nothing more than the decoded current "
      "state.",
      md(rows, ["Contrast", "MIMIC-IV test, Δ AUROC (95% CI)", "eICU, Δ AUROC (95% CI)"]))

# ---- Table 9: horizon ---------------------------------------------------------
MH = pd.read_csv(PRED / "multihorizon_metrics.csv")
HC = pd.read_csv(PRED / "horizon_contrasts.csv")
FX = pd.read_csv(PRED / "horizon_sensitivity.csv")
fx = FX[FX.analysis == "population fixed at 24 h"]

rows = []
SEQ = pd.read_csv(PRED / "metrics_L6.csv")
SC = pd.read_csv(PRED / "sequence_contrasts.csv")
for h in [6, 12, 24]:
    e = MH[(MH.scope == "eicu-full") & (MH.horizon_h == h)].set_index("model")
    q = SEQ[(SEQ.scope == "eicu-full") & (SEQ.horizon_h == h)].iloc[0]
    sc = SC[(SC.scope == "eicu-full") & (SC.horizon_h == h)].iloc[0]
    win = e.loc[["L3", "L4", "L5"], "change_auroc"].max()
    share = (e.loc["L2", "change_auroc"] - 0.5) / (q.change_auroc - 0.5) * 100
    f = fx[(fx.scope == "eicu-full") & (fx.horizon_h == h)].iloc[0]
    rows.append([f"{h} h", f"{int(e.loc['L2','pairs']):,}",
                 f"{e.loc['L2','changed_pct']:.1f}",
                 f"{e.loc['L2','change_auroc']:.3f}", f"{win:.3f}",
                 f"{q.change_auroc:.3f}",
                 f"+{sc.delta_auroc:.3f} ({sc.lo:+.3f} to {sc.hi:+.3f})",
                 f"{share:.1f}", f"{f['L2']:.3f}"])

T9 = ("**Table 9.** Discrimination as a function of how far ahead the state is predicted, "
      "eICU. The frozen transition matrix is raised to the corresponding power; nothing else "
      "changes for the unfitted predictors, and the fitted predictors are refitted on the "
      "MIMIC-IV training split at each horizon. Discrimination does not decay with the "
      "horizon but rises, and two quantities move with it that must be read together: the "
      "event becomes more common and the eligible set contracts to stays lasting that long. "
      "The last column repeats the frozen model with the population held fixed at the "
      "24-hour eligible set, so that the same patients and index windows are used at all "
      "three horizons and only the target moves; the rise persists, and is slightly steeper, "
      "so it is not produced by that contraction. The ceiling is set by the sequence model, "
      "which reads the observation history directly; the single-window models are shown "
      "alongside because they cannot reach it. The frozen share is the discrimination above "
      "the null reached with no fitting at all, as a percentage of that reached by the "
      "sequence model, and it rises with the horizon: what the four states discard is "
      "concentrated at short range.",
      md(rows, ["Horizon", "Pairs", "Changed, %", "Frozen HMM, filtered",
                "Best single-window model", "Sequence model",
                "Sequence − frozen HMM (95% CI)", "Frozen share, %",
                "Frozen HMM, population fixed"]))

(OUT / "tables_aim4.md").write_text(
    "# Aim 4 tables\n\n" + f"{T7[0]}\n\n{T7[1]}\n\n\n{T8[0]}\n\n{T8[1]}\n\n\n{T9[0]}\n\n{T9[1]}\n")
print(f"wrote {OUT/'tables_aim4.md'}")

# the sentence the results section turns on: the ceiling is the sequence model
for sc in ["mimic-test", "eicu-full"]:
    for h in [6, 12, 24]:
        l2 = MH[(MH.scope == sc) & (MH.horizon_h == h) & (MH.model == "L2")].change_auroc.iloc[0]
        l6 = SEQ[(SEQ.scope == sc) & (SEQ.horizon_h == h)].change_auroc.iloc[0]
        print(f"  {sc:<12} {h:>2}h  frozen {l2:.3f} / sequence {l6:.3f}  ->  "
              f"frozen supplies {(l2-0.5)/(l6-0.5)*100:.1f}% of the discrimination above the null")
