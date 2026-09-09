"""
Paper 2: the four main-text tables for the JAMIA submission.

JAMIA allows four tables in Research and Applications. The nine tables of the full
manuscript are reduced by keeping the four that carry a claim the text cannot make
without them, and moving the rest to the supplement:

  Table 1  cohort characteristics
  Table 2  frozen-model transport against the five prespecified criteria
  Table 3  state-outcome associations
  Table 4  prediction, in two panels: the ladder at 6 h and the three horizons

Variable availability, the GCS imputation sensitivity, the between-hospital
variance components and the paired contrasts move to the supplement; their
headline numbers are stated in the text instead.

Generated from the same pipeline outputs as the full manuscript, so the two
cannot disagree.

Output: manuscript/tables_jamia.md
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "manuscript"
PRED = HERE / "prediction"
P = "neurologically preserved-low support"
R = "neurological impairment-respiratory support"
N = "neurological impairment-renal dysfunction"
L = "neurological impairment-low support"
ORDER = [P, L, N, R]
SHORT = {P: "Neurologically preserved–low support", L: "Neurological impairment–low support",
         N: "Neurological impairment–renal dysfunction", R: "Neurological impairment–respiratory support"}


def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] +
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


# reuse the full manuscript's own table text so the two cannot diverge
full = (OUT / "tables_all.md").read_text()


def lift(n):
    """Take table n from the full set, caption and body, unchanged."""
    i = full.index(f"**Table {n}.**")
    j = full.find("**Table ", i + 10)
    return full[i:(j if j != -1 else len(full))].strip()


def renumber(block, new):
    old = block[len("**Table "):block.index(".")]
    return block.replace(f"**Table {old}.**", f"**Table {new}.**", 1)


blocks = [renumber(lift(1), 1), renumber(lift(3), 2), renumber(lift(5), 3)]

# ---- Table 4: prediction, two panels ------------------------------------------
M = pd.read_csv(PRED / "metrics_by_model.csv")
L6 = pd.read_csv(PRED / "metrics_L6.csv")
MH = pd.read_csv(PRED / "multihorizon_metrics.csv")
SC = pd.read_csv(PRED / "sequence_contrasts.csv")
mt = M[M.scope == "mimic-test"].set_index("model")
ef = M[M.scope == "eicu-full"].set_index("model")
NAME = {"L0": "Persistence (null)", "L1": "Frozen transition matrix",
        "L2": "Frozen HMM, forward filtered", "L3": "Multinomial logistic",
        "L4": "Gradient boosting", "L5": "Gradient boosting + covariates"}
INFO = {"L0": "current state, assumed to continue", "L1": "decoded current state",
        "L2": "full observation history through *t*", "L3": "current state + 21 variables at *t*",
        "L4": "as L3, without a linearity assumption", "L5": "as L4 + age, sex, subtype, window index"}
FIT = {"L0": "no", "L1": "no", "L2": "no", "L3": "yes", "L4": "yes", "L5": "yes"}


def ci(d, c="change_auroc"):
    return f"{d[c]:.3f} ({d[f'{c}_lo']:.3f}–{d[f'{c}_hi']:.3f})"


rows = [[NAME[k], INFO[k], FIT[k], ci(mt.loc[k]), ci(ef.loc[k]),
         f"{ef.loc[k,'change_sens_at_90spec']*100:.1f}"] for k in NAME]
l6m = L6[(L6.scope == "mimic-test") & (L6.horizon_h == 6)].iloc[0]
l6e = L6[(L6.scope == "eicu-full") & (L6.horizon_h == 6)].iloc[0]
rows.append(["Sequence model (GRU)", "the raw observation sequence through *t*", "yes",
             ci(l6m), ci(l6e), f"{l6e.change_sens_at_90spec*100:.1f}"])
panel_a = md(rows, ["Predictor", "Information used", "Fitted",
                    "MIMIC-IV test, AUROC (95% CI)", "eICU, AUROC (95% CI)",
                    "eICU sens. at 90% spec., %"])

rows = []
# The caption quotes two of these; taking them from the same strings the table
# prints makes it impossible for caption and cell to round differently.
SENS_BY_H = {}
for h in [6, 12, 24]:
    e = MH[(MH.scope == "eicu-full") & (MH.horizon_h == h)].set_index("model")
    q = L6[(L6.scope == "eicu-full") & (L6.horizon_h == h)].iloc[0]
    sc = SC[(SC.scope == "eicu-full") & (SC.horizon_h == h)].iloc[0]
    share = (e.loc["L2", "change_auroc"] - 0.5) / (q.change_auroc - 0.5) * 100
    # AUROC summarises every threshold at once; a clinician uses one. Without a
    # sensitivity column the panel cannot say whether a longer horizon is more
    # usable or merely easier to rank.
    sens = f"{e.loc['L2','change_sens_at_90spec']*100:.1f} / {q.change_sens_at_90spec*100:.1f}"
    SENS_BY_H[h] = sens.split(" / ")[0]
    rows.append([f"{h} h", f"{int(e.loc['L2','pairs']):,}", f"{e.loc['L2','changed_pct']:.1f}",
                 f"{e.loc['L2','change_auroc']:.3f}", f"{q.change_auroc:.3f}",
                 f"+{sc.delta_auroc:.3f} ({sc.lo:+.3f} to {sc.hi:+.3f})", f"{share:.1f}", sens])
panel_b = md(rows, ["Horizon", "Pairs", "Changed, %", "Frozen HMM", "Sequence model",
                    "Sequence − frozen (95% CI)", "Retained, %",
                    "Sens. at 90% spec., %"])

cap = ("**Table 4.** Prediction of the next state. The event is that the state differs in "
       "the following window; it occurs in 9.3% of eICU pairs, so a rule predicting no "
       "change scores 0.906 accuracy while identifying nobody, and discrimination for "
       "change is used instead. *Panel A, a 6-hour horizon:* predictors ordered by the "
       "information each is permitted to use. The first three involve no fitting and are "
       "the discovery model used for a different operation; the rest were fitted on the "
       "MIMIC-IV training split and frozen before eICU was decoded. The sequence model is "
       "included as an empirical benchmark and is the mean of three seeds. *Panel B, eICU "
       "across "
       "horizons:* the frozen transition matrix raised to the corresponding power; the fitted "
       "predictors were trained separately in the MIMIC-IV training split for each horizon and "
       "frozen before external evaluation, never refitted in eICU. "
       "Discrimination rises rather than decays, and the proportion retained — the "
       "discrimination above chance reached with no fitting, as a percentage of the sequence "
       "model's, a fraction of discrimination and not of information — rises with it, so what "
       "the four states discard is concentrated at short range. "
       "The final column gives sensitivity at the threshold yielding 90% specificity, "
       "as frozen model / sequence model. It is reported because area "
       "under the curve summarises every threshold at once whereas a clinician uses one: "
       f"the frozen model identifies {SENS_BY_H[6]}% of transitions at six hours and "
       f"{SENS_BY_H[24]}% at "
       "twenty-four, so the longer horizons are more usable and not merely easier to "
       "rank, and none of the three reaches a sensitivity that would support alerting on "
       "an individual patient. "
       "Intervals are patient-clustered bootstrap percentiles; the horizon contrasts are "
       "paired within the same resample. Full results across all scopes are in "
       "Supplementary Table S4.")

blocks.append(f"{cap}\n\n{panel_a}\n\n{panel_b}")
(OUT / "tables_jamia.md").write_text("# Tables\n\n" + "\n\n\n".join(blocks) + "\n")
print(f"wrote {OUT/'tables_jamia.md'}")
for i, b in enumerate(blocks, 1):
    n = len([l for l in b.splitlines() if l.startswith("|")])
    print(f"  Table {i}: {n} table lines")
