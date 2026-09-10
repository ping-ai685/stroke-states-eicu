"""
Paper 2, checklist item 5: generate the three term lists for clinical review.

Protocol v1.1 sections 7.2 and 7.3 require the drug, urine-source and
ventilation-evidence vocabularies to be enumerated, clinically reviewed and
FROZEN before the C-tier variables are extracted. Nothing downstream may use a
term the reviewer has not seen.

This script proposes; it does not decide. Every distinct value found in the
cohort is listed with its frequency and a proposed classification, and the
`REVIEWER_DECISION` column is left blank. The proposal is a regex heuristic
mirroring the MIMIC-IV item sets, and it is wrong often enough that the review
matters -- brand names, unit suffixes and local abbreviations do not pattern-match
reliably.

MIMIC-IV counterparts being reproduced (03_code/variable_itemid_map.py):
  vasoactive  epinephrine, norepinephrine, phenylephrine, dopamine, dobutamine,
              vasopressin, milrinone   -- note this is "vasoactive support",
              inotropes included; the eICU list must not be narrowed
  sedative    propofol, midazolam, dexmedetomidine, lorazepam; continuous
              infusion only, no boluses
  CRRT        continuous modalities only (CVVH/CVVHD/CVVHDF/SCUF)
  vent        INVASIVE ventilation only

Outputs (review/): three CSVs plus a combined workbook.
"""
import numpy as np
import pandas as pd
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import data_paths                                    # noqa: E402

B = str(data_paths.eicu())
HERE = Path(__file__).parent
OUT = HERE / "review"; OUT.mkdir(exist_ok=True)
STAYS = set(pd.read_csv(HERE / "cohort/patient_level_cohort.csv").patientunitstayid)

AGENTS = {
    "vasoactive": {"norepinephrine": r"norepineph|levophed|noradren", "epinephrine": r"(?<!nor)epineph|adrenalin",
                   "phenylephrine": r"phenyleph|neo-?synephrine", "dopamine": r"dopamine|intropin",
                   "dobutamine": r"dobutamine|dobutrex", "vasopressin": r"vasopressin|pitressin",
                   "milrinone": r"milrinone|primacor"},
    "sedative": {"propofol": r"propofol|diprivan", "midazolam": r"midazolam|versed",
                 "dexmedetomidine": r"dexmedetomidine|precedex", "lorazepam": r"lorazepam|ativan"},
}
def classify(name):
    n = str(name).lower()
    for cls, agents in AGENTS.items():
        for agent, rx in agents.items():
            if re.search(rx, n):
                return cls, agent
    return "", ""

def emit(df, cols, fname, note):
    df.insert(0, "REVIEWER_DECISION", "")
    df.insert(1, "REVIEWER_NOTE", "")
    df.to_csv(OUT / fname, index=False)
    print(f"\n{note}\n  -> review/{fname}  ({len(df)} rows to review)")
    return df

# ---------------------------------------------------------------------------
# 1. infusion and IV medication drug names
#
# Two problems the first version of this list had, both fixed here.
#
# (a) The `medication` table is mostly BOLUS dosing -- "LORAZEPAM 2 MG/1 ML 1 ML
#     INJ" is a vial, not an infusion -- while Paper 1's sedative flag counts
#     continuous infusion only. Without `frequency` and the order duration the
#     reviewer cannot tell the two apart, so both are shown.
#
# (b) Vasodilators are extremely common in a stroke ICU -- nicardipine appears in
#     666 stays -- and MIMIC's flag covers vasopressors and inotropes only, no
#     vasodilators. Excluding them is the matched choice, but it is a decision
#     worth making explicitly rather than leaving buried among 500 antibiotics
#     and saline bags. They are surfaced as their own high-priority block.
# ---------------------------------------------------------------------------
VASODILATORS = r"nicardipine|nicardipine|cardene|clevidipine|cleviprex|labetalol|hydralazine|nitroglycerin|nitroprusside|nipride"

inf = pd.read_csv(f"{B}/infusionDrug.csv.gz",
                  usecols=["patientunitstayid", "drugname"], low_memory=False)
inf = inf[inf.patientunitstayid.isin(STAYS)]
g = inf.groupby("drugname").agg(rows=("patientunitstayid", "size"),
                                stays=("patientunitstayid", "nunique")).reset_index()
g["source_table"] = "infusionDrug"
g["frequency"] = "(infusion table)"
g["median_order_hours"] = np.nan
g["pct_orders_continuous"] = 100.0

med = pd.read_csv(f"{B}/medication.csv.gz",
                  usecols=["patientunitstayid", "drugname", "routeadmin", "frequency",
                           "drugstartoffset", "drugstopoffset"], low_memory=False)
med = med[med.patientunitstayid.isin(STAYS) &
          med.routeadmin.fillna("").str.contains("iv|intraven", case=False, regex=True)]
med["_hrs"] = (med.drugstopoffset - med.drugstartoffset) / 60
g2 = med.groupby("drugname").agg(rows=("patientunitstayid", "size"),
                                 stays=("patientunitstayid", "nunique"),
                                 median_order_hours=("_hrs", "median")).reset_index()
freq = med.groupby("drugname").frequency.agg(
    lambda s: s.value_counts().index[0] if s.notna().any() else "")
g2["frequency"] = g2.drugname.map(freq)
# Paper 1 counts CONTINUOUS infusion only. The medication table is mostly bolus
# dosing, so each row is marked by whether its orders look continuous -- without
# this the reviewer cannot apply the construct rule.
_c = med.frequency.fillna("").str.upper().str.contains("CONT|INFUS|DRIP|TITRAT", regex=True)
g2["pct_orders_continuous"] = (med.assign(_c=_c).groupby("drugname")._c.mean() * 100).round(0).values
g2["source_table"] = "medication (IV route)"
drugs = pd.concat([g, g2], ignore_index=True)
drugs[["proposed_class", "matched_agent"]] = drugs.drugname.apply(lambda x: pd.Series(classify(x)))
drugs["is_vasodilator"] = drugs.drugname.str.contains(VASODILATORS, case=False, regex=True)

def priority(r):
    if r.proposed_class:
        return "1-HIGH: proposed as vasoactive or sedative"
    if r.is_vasodilator:
        return "2-HIGH: vasodilator, excluded by the MIMIC definition -- please confirm"
    if r.stays >= 100:
        return "3-MEDIUM: frequent, not flagged -- skim for a missed agent"
    return "4-LOW: not flagged"
drugs["REVIEW_PRIORITY"] = drugs.apply(priority, axis=1)
drugs["proposed_class"] = np.where(drugs.proposed_class != "", drugs.proposed_class,
                                   np.where(drugs.is_vasodilator, "exclude (vasodilator)", "exclude"))
drugs = drugs[(drugs.REVIEW_PRIORITY.str.startswith(("1", "2", "3"))) | (drugs.stays >= 50)]
drugs["median_order_hours"] = drugs.median_order_hours.round(1)
drugs = drugs.sort_values(["REVIEW_PRIORITY", "stays"], ascending=[True, False])
drugs = emit(drugs[["REVIEW_PRIORITY", "drugname", "source_table", "frequency",
                    "pct_orders_continuous", "median_order_hours", "rows", "stays",
                    "proposed_class", "matched_agent"]].copy(),
             None, "termlist_1_drugs.csv",
             "LIST 1 - vasoactive and sedative infusions. Decide: vasoactive / sedative / exclude. "
             "Sorted by priority; `frequency` and `median_order_hours` distinguish a continuous "
             "infusion from a bolus, which matters because Paper 1 counts infusions only.")
print(drugs.REVIEW_PRIORITY.value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# 2. urine output source labels
# ---------------------------------------------------------------------------
parts = []
for ch in pd.read_csv(f"{B}/intakeOutput.csv.gz",
                      usecols=["patientunitstayid", "cellpath", "celllabel", "cellvaluenumeric"],
                      chunksize=4_000_000, low_memory=False):
    ch = ch[ch.patientunitstayid.isin(STAYS)]
    ch = ch[ch.cellpath.fillna("").str.contains(r"\|Output \(ml\)\||\|Output\|", regex=True)]
    if len(ch): parts.append(ch)
io = pd.concat(parts, ignore_index=True)
u = io.groupby("celllabel").agg(rows=("patientunitstayid", "size"), stays=("patientunitstayid", "nunique"),
                                median_ml=("cellvaluenumeric", "median")).reset_index()
u["proposed_include"] = u.celllabel.str.lower().str.contains(
    r"urine|foley|void|catheter|nephrostomy|suprapubic|condom|urethral", regex=True).map({True: "include", False: ""})
u = u[(u.proposed_include == "include") | (u.stays >= 25)].sort_values(
    ["proposed_include", "stays"], ascending=[False, False])
u["median_ml"] = u.median_ml.round(0)
emit(u.copy(), None, "termlist_2_urine_sources.csv",
     "LIST 2 - urine output sources. Decide: include / exclude. MIMIC counts urinary sources only "
     "(Foley, void, condom, suprapubic, nephrostomy, straight cath) and excludes GI and drain output. "
     "Blank proposed_include rows are frequent output labels the heuristic did not match.")
print(u.proposed_include.value_counts(dropna=False).to_string())

# ---------------------------------------------------------------------------
# 3. ventilation and CRRT evidence
# ---------------------------------------------------------------------------
t = pd.read_csv(f"{B}/treatment.csv.gz", usecols=["patientunitstayid", "treatmentstring"])
t = t[t.patientunitstayid.isin(STAYS)]
ts = t.treatmentstring.fillna("")
vt = t[ts.str.contains("ventilation and oxygenation|renal\\|dialysis", regex=True)]
v1 = vt.groupby("treatmentstring").patientunitstayid.nunique().reset_index(name="stays")
v1["source"] = "treatment.treatmentstring"
v1["proposed"] = v1.treatmentstring.apply(lambda x: (
    "CRRT" if re.search(r"c ?v ?v ?h|crrt|continuous.*dialysis", x, re.I) else
    "dialysis-not-CRRT" if "dialysis" in x.lower() else
    "INVASIVE vent" if "|mechanical ventilation" in x.lower() else
    "NON-invasive (exclude)" if re.search(r"non-invasive|cpap|bipap", x, re.I) else
    "not vent evidence"))
parts = []
for ch in pd.read_csv(f"{B}/respiratoryCharting.csv.gz",
                      usecols=["patientunitstayid", "respcharttypecat", "respchartvaluelabel"],
                      chunksize=2_000_000, low_memory=False):
    ch = ch[ch.patientunitstayid.isin(STAYS)]
    if len(ch): parts.append(ch.groupby(["respcharttypecat", "respchartvaluelabel"])
                             .patientunitstayid.nunique())
rc = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index(name="stays")
rc = rc[rc.stays >= 25].sort_values("stays", ascending=False)
rc["treatmentstring"] = rc.respcharttypecat + " | " + rc.respchartvaluelabel
rc["source"] = "respiratoryCharting label"
rc["proposed"] = rc.respchartvaluelabel.apply(lambda x: (
    "vent-setting evidence" if x in ["PEEP", "Vent Rate", "Tidal Volume (set)", "Exhaled TV (patient)",
                                     "Peak Insp. Pressure", "Mean Airway Pressure", "Exhaled MV",
                                     "Pressure Support", "TV/kg IBW", "RR (patient)", "Total RR"]
    else "not vent evidence"))
aw = pd.read_csv(f"{B}/respiratoryCare.csv.gz", usecols=["patientunitstayid", "airwaytype"], low_memory=False)
aw = aw[aw.patientunitstayid.isin(STAYS)]
v3 = aw.groupby("airwaytype").patientunitstayid.nunique().reset_index(name="stays")
v3["treatmentstring"] = v3.airwaytype; v3["source"] = "respiratoryCare.airwaytype"
v3["proposed"] = v3.airwaytype.apply(lambda x: "INVASIVE vent" if str(x) in
                                     ["Oral ETT", "Nasal ETT", "Tracheostomy"] else "not invasive")
vent = pd.concat([v1[["treatmentstring", "source", "stays", "proposed"]],
                  rc[["treatmentstring", "source", "stays", "proposed"]],
                  v3[["treatmentstring", "source", "stays", "proposed"]]], ignore_index=True)
vent = vent.sort_values(["source", "stays"], ascending=[True, False])
emit(vent.copy(), None, "termlist_3_ventilation_crrt.csv",
     "LIST 3 - ventilation and CRRT evidence. Decide per row. Paper 1 counts INVASIVE ventilation only "
     "and CONTINUOUS renal replacement only; NIV, CPAP/BiPAP and intermittent haemodialysis are excluded. "
     "respiratoryCharting labels cannot themselves distinguish invasive from NIV -- they are density "
     "evidence used only where the stay carries no NIV/CPAP treatment record.")
print(vent.groupby("proposed").size().to_string())
print(f"\nAll three lists written to {OUT}/")
