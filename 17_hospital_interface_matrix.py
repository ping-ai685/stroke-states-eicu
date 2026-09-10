"""
Paper 2, checklist item 5d: hospital-level interface availability matrix, and the
ventilation ascertainment rule.

PI decision, 23 August 2026. Mechanical ventilation must not be resolved at the
patient level. Setting all 6,146 undetermined stays to missing throws away real
negatives; keeping only the 2,133 determinable stays enriches for ventilated
patients severely -- 87.8% of that subset is ventilated. The correct level is the
hospital interface, which is also what protocol v1.1 section 8 already says about
structural missingness.

  - In a hospital whose respiratory interfaces demonstrably work, "no invasive
    evidence" is a real zero.
  - In a hospital where the relevant interface is essentially absent, ventilation
    is STRUCTURALLY UNASCERTAINABLE, and that hospital does not enter the primary
    full 21-variable frozen-model validation.

Blind by construction: no HMM, no state assignment, no outcome field is read.

A stroke ICU with a reasonable number of patients and not a single invasive
ventilation record is far more likely to have an unwired interface than a ward
where nobody was ever intubated, so the rule requires a hospital to demonstrate
that its interface can produce positives.

Outputs (audit/): hospital_interface_matrix.csv, ventilation_ascertainment_tiers.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import data_paths                                    # noqa: E402

B = str(data_paths.eicu())
HERE = Path(__file__).parent
OUT = HERE / "audit"

coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv",
                  usecols=["patientunitstayid", "hospitalid", "n_windows", "unittype"])
STAYS = set(coh.patientunitstayid)
vent = pd.read_csv(HERE / "review_returned/termlist_3_ventilation_crrt_REVIEWED.csv", encoding="utf-8-sig")
INV_T = set(vent.loc[(vent.REVIEWER_DECISION == "INVASIVE vent") & (vent.source == "treatment.treatmentstring"), "treatmentstring"])
INV_A = set(vent.loc[(vent.REVIEWER_DECISION == "INVASIVE vent") & (vent.source == "respiratoryCare.airwaytype"), "treatmentstring"])
NIV_T = set(vent.loc[(vent.REVIEWER_DECISION == "NON-invasive (exclude)") & (vent.source == "treatment.treatmentstring"), "treatmentstring"])

def stayset(df, col=None, vals=None):
    d = df[df[col].isin(vals)] if col else df
    return set(d.patientunitstayid)

t = pd.read_csv(f"{B}/treatment.csv.gz", usecols=["patientunitstayid", "treatmentstring"])
t = t[t.patientunitstayid.isin(STAYS)]
rc = pd.read_csv(f"{B}/respiratoryCare.csv.gz",
                 usecols=["patientunitstayid", "airwaytype", "ventstartoffset"], low_memory=False)
rc = rc[rc.patientunitstayid.isin(STAYS)]
rch = set()
for ch in pd.read_csv(f"{B}/respiratoryCharting.csv.gz", usecols=["patientunitstayid"],
                      chunksize=4_000_000, low_memory=False):
    rch |= set(ch.patientunitstayid) & STAYS
inf = pd.read_csv(f"{B}/infusionDrug.csv.gz", usecols=["patientunitstayid"], low_memory=False)
med = pd.read_csv(f"{B}/medication.csv.gz", usecols=["patientunitstayid", "routeadmin"], low_memory=False)

SETS = {
    "any_treatment_row": stayset(t),
    "treatment_vent_branch": stayset(t[t.treatmentstring.str.contains("ventilation and oxygenation", na=False)]),
    "treatment_invasive": stayset(t, "treatmentstring", INV_T),
    "treatment_niv": stayset(t, "treatmentstring", NIV_T),
    "respcare_any": stayset(rc),
    "respcare_airwaytype": stayset(rc[rc.airwaytype.notna()]),
    "respcare_airway_invasive": stayset(rc, "airwaytype", INV_A),
    "respcharting_any": rch,
    "infusionDrug_any": stayset(inf),
    "medication_iv_any": stayset(med[med.routeadmin.fillna("").str.contains("iv|intraven", case=False, regex=True)]),
}
m = coh[["patientunitstayid", "hospitalid"]].copy()
for k, v in SETS.items():
    m[k] = m.patientunitstayid.isin(v).astype(int)
m["invasive_determinable"] = ((m.treatment_invasive | m.respcare_airway_invasive) |
                              (m.treatment_niv)).astype(int)
m["invasive_positive"] = (m.treatment_invasive | m.respcare_airway_invasive).astype(int)

h = m.groupby("hospitalid").agg(n_patients=("patientunitstayid", "size"),
                                **{k: (k, "mean") for k in SETS},
                                pct_determinable=("invasive_determinable", "mean"),
                                pct_invasive_positive=("invasive_positive", "mean"),
                                n_invasive_positive=("invasive_positive", "sum")).reset_index()
h = h.merge(coh.groupby("hospitalid").n_windows.sum().reset_index(), on="hospitalid")
for c in list(SETS) + ["pct_determinable", "pct_invasive_positive"]:
    h[c] = (h[c] * 100).round(1)
h.sort_values("n_patients", ascending=False).to_csv(OUT / "hospital_interface_matrix.csv", index=False)

print(f"=== interface coverage across {len(h)} hospitals (share of the hospital's stroke stays) ===")
print(h[list(SETS)].describe().loc[["mean", "50%", "min", "max"]].round(1).T.to_string())
print(f"\nhospitals with ZERO invasive-ventilation records: {(h.n_invasive_positive == 0).sum()} "
      f"of {len(h)}, holding {h.loc[h.n_invasive_positive == 0, 'n_patients'].sum():,} patients")
z = h[(h.n_invasive_positive == 0) & (h.n_patients >= 20)]
print(f"  of those, {len(z)} have >=20 stroke patients and still zero -- "
      f"implausible clinically, so almost certainly an unwired interface")

# ---- candidate ascertainment tiers ------------------------------------------
w = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv").merge(
    coh[["patientunitstayid", "hospitalid"]], on="patientunitstayid")
rows = []
CANDS = [
    ("no filter (current state)", h.hospitalid),
    ("A: >=1 invasive record", h.loc[h.n_invasive_positive >= 1, "hospitalid"]),
    ("B: >=1 invasive AND >=20 patients", h.loc[(h.n_invasive_positive >= 1) & (h.n_patients >= 20), "hospitalid"]),
    ("C: >=1 invasive AND determinable >=25%", h.loc[(h.n_invasive_positive >= 1) & (h.pct_determinable >= 25), "hospitalid"]),
    ("D: >=1 invasive AND determinable >=50%", h.loc[(h.n_invasive_positive >= 1) & (h.pct_determinable >= 50), "hospitalid"]),
    ("E: D AND >=20 patients", h.loc[(h.n_invasive_positive >= 1) & (h.pct_determinable >= 50) & (h.n_patients >= 20), "hospitalid"]),
]
for name, ids in CANDS:
    ids = set(ids)
    sub = w[w.hospitalid.isin(ids)]
    pats = coh[coh.hospitalid.isin(ids)]
    rows.append({"rule": name, "hospitals": len(ids), "patients": len(pats),
                 "pct_of_cohort": round(len(pats) / len(coh) * 100),
                 "windows": len(sub),
                 "mech_vent_pct_windows": round(sub.mech_vent.mean() * 100, 1),
                 "sedative_pct_windows": round(sub.sedative.mean() * 100, 1),
                 "vasoactive_pct_windows": round(sub.vasopressor.mean() * 100, 1)})
tiers = pd.DataFrame(rows)
tiers["mimic_mech_vent"] = 24.8
tiers.to_csv(OUT / "ventilation_ascertainment_tiers.csv", index=False)
print("\n=== candidate ascertainment rules (MIMIC targets: vent 24.8, sedative 19.7, vasoactive 9.3) ===")
print(tiers.to_string(index=False))
print("\nNote: `vasopressor` is the Paper 1 COLUMN name and is kept for feature-order")
print("alignment with the frozen model; the variable is VASOACTIVE SUPPORT and includes")
print("dobutamine and milrinone. Reported labels say vasoactive support throughout.")
