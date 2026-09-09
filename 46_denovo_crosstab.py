"""
Paper 2: where the frozen states' windows land when eICU is refitted from scratch.

The correspondence tables report, for each de novo state, which discovery state it
most resembles. They do not report the converse: for each discovery state, where
its windows go. That converse is what distinguishes two explanations of why
neurological impairment-low support is not recovered.

  If it were a residual class with no signal, its windows would scatter.
  If it lies across a boundary, they would go somewhere specific.

They go somewhere specific, and the same place under both K, which is why the
Discussion describes the state as lying across a boundary rather than as a
residual.

Output: manuscript/denovo_crosstab.md, appended to the supplementary tables.
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "manuscript"
SHORT = {"neurologically preserved-low support": "Preserved, no support",
         "neurological impairment-respiratory support": "Impaired, respiratory support",
         "neurological impairment-renal dysfunction": "Impaired, renal dysfunction",
         "neurological impairment-low support": "Impaired, no support"}
ORDER = list(SHORT.values())

frozen = pd.read_csv(HERE / "results_full_model/eicu_state_assignments_full_A1.csv")[
    ["patientunitstayid", "window_idx", "state"]]


def crosstab(assign_file, state_col, match_file, k):
    a = pd.read_csv(HERE / "results_denovo" / assign_file)
    m = frozen.merge(a, on=["patientunitstayid", "window_idx"])
    m["frozen"] = m.state.map(SHORT)
    mt = pd.read_csv(HERE / "results_denovo" / match_file).set_index("eicu_state")
    # A 55-character column header wraps to three lines and makes the table
    # unreadable. The full state names are in the row labels and in the legend;
    # here one word is enough to say which discovery state each column resembles.
    TERSE = {"Preserved, no support": "preserved",
             "Impaired, respiratory support": "respiratory",
             "Impaired, renal dysfunction": "renal",
             "Impaired, no support": "impaired, no support"}
    lab = {i: f"de novo {i} (≈{TERSE[SHORT[mt.loc[i,'best_match_state']]]}, "
              f"r {mt.loc[i,'best_match_correlation']:.2f})" for i in mt.index}
    m["denovo"] = m[state_col].map(lab)
    ct = (pd.crosstab(m["frozen"], m["denovo"], normalize="index") * 100).reindex(ORDER)
    return ct.round(1), len(m)


def md(ct, caption):
    hdr = ["Frozen state"] + list(ct.columns)
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for idx, row in ct.iterrows():
        lines.append("| " + " | ".join([idx] + [f"{v:.1f}" for v in row]) + " |")
    return caption + "\n\n" + "\n".join(lines)


ct3, n3 = crosstab("eicu_denovo_K3_assignments.csv", "k3_state", "denovo_K3_matching.csv", 3)
ct4, n4 = crosstab("eicu_denovo_assignments.csv", "denovo_state", "denovo_K4_matching.csv", 4)

cap = ("**Supplementary Table S5.** Where the windows of each frozen state are placed when a "
       "model is fitted to eICU from scratch. Rows are the four discovery states as assigned by "
       "the frozen model; columns are the states eICU recovers on its own; cells are row "
       "percentages, so each row sums to 100. The correspondence tables reported elsewhere read "
       "in the other direction — for each de novo state, which discovery state it resembles — and "
       "cannot show what happens to a discovery state that is not recovered. Three of the four "
       "discovery states are placed almost intact. The windows of neurological impairment–low "
       "support are not: two thirds are placed with the respiratory-support state under both "
       "solutions, and only 13.0% (three-state) and 20.6% (four-state) with the preserved state. "
       "They are therefore not scattered, which a class with no signal would be; they are "
       "consistently assigned along the neurological axis, to states whose patients are receiving "
       "organ support that these patients are not.")

body = (f"{cap}\n\n"
        f"*Three-state solution (eICU's most reproducible; {n3:,} windows in the strictest scope)*\n\n"
        + md(ct3, "").strip() + "\n\n"
        f"*Four-state solution ({n4:,} windows)*\n\n"
        + md(ct4, "").strip() + "\n")

(OUT / "denovo_crosstab.md").write_text(body)
print(f"wrote {OUT/'denovo_crosstab.md'}\n")
print("K=3:"); print(ct3.to_string()); print()
print("K=4:"); print(ct4.to_string())
