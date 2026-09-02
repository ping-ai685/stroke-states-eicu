"""
Paper 2, checklist item 5 CLOSED: freeze the term dictionary and the ventilation
ascertainment list as v1.0.

Everything below was adjudicated clinically (item 5b), technically checked
(item 5c) and decided by the PI (23 Aug 2026). After this script runs, no
downstream code may use a term that is not in these files, and the extraction
fails loudly if eICU presents one.

Ventilation ascertainment, PI-approved rule N3: a hospital is structurally
unascertainable for ventilation if BOTH determining interfaces -- the `treatment`
ventilation branch and `respiratoryCare.airwaytype` -- cover under 10% of its
stroke stays. Justified on wiring alone. The tempting alternative that reproduces
MIMIC's 24.8% exactly was rejected because its threshold sits at the median of a
unimodal distribution and is justified only by hitting the transport target,
which protocol section 8.2 forbids.

Outputs (frozen_dictionary_v1.0/).
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "review_returned"
OUT = HERE / "frozen_dictionary_v1.0"; OUT.mkdir(exist_ok=True)
FREEZE = "2026-08-23"

dr = pd.read_csv(R / "termlist_1_drugs_REVIEWED.csv", encoding="utf-8-sig")
ur = pd.read_csv(R / "termlist_2_urine_sources_REVIEWED.csv", encoding="utf-8-sig")
vt = pd.read_csv(R / "termlist_3_ventilation_crrt_REVIEWED.csv", encoding="utf-8-sig")

def freeze(df, decision_col, name_col, keep, fname, variable):
    d = df[df[decision_col].isin(keep)][[name_col, decision_col, "REVIEWER_NOTE"]].copy()
    d.columns = ["term", "class", "adjudication_note"]
    d.insert(0, "variable", variable)
    d["freeze_date"] = FREEZE
    d.to_csv(OUT / fname, index=False)
    return d

a = freeze(dr, "REVIEWER_DECISION", "drugname", ["vasoactive", "sedative"],
           "terms_drugs.csv", "vasoactive_support / sedative")
b = freeze(ur, "REVIEWER_DECISION", "celllabel", ["include"],
           "terms_urine_sources.csv", "urine_output_ml")
c = freeze(vt, "REVIEWER_DECISION", "treatmentstring",
           ["INVASIVE vent", "NON-invasive (exclude)", "vent-setting evidence", "CRRT"],
           "terms_ventilation_crrt.csv", "mech_vent / crrt")
# the ventilation file needs its source column to be usable
c2 = vt[vt.REVIEWER_DECISION.isin(["INVASIVE vent", "NON-invasive (exclude)",
                                   "vent-setting evidence", "CRRT"])].copy()
c2 = c2[["treatmentstring", "source", "REVIEWER_DECISION", "REVIEWER_NOTE"]]
c2.columns = ["term", "source_field", "class", "adjudication_note"]
c2["freeze_date"] = FREEZE
c2.to_csv(OUT / "terms_ventilation_crrt.csv", index=False)

# ---- ventilation ascertainment, rule N3 -------------------------------------
h = pd.read_csv(HERE / "audit/hospital_interface_matrix.csv")
h["vent_ascertainable"] = ~((h.treatment_vent_branch < 10) & (h.respcare_airwaytype < 10))
h["exclusion_reason"] = h.vent_ascertainable.map(
    {True: "", False: "both invasive-determining interfaces cover <10% of stays"})
h["rule"] = "N3, frozen " + FREEZE
h[["hospitalid", "n_patients", "treatment_vent_branch", "respcare_airwaytype",
   "pct_determinable", "pct_invasive_positive", "vent_ascertainable",
   "exclusion_reason", "rule"]].sort_values(["vent_ascertainable", "n_patients"],
   ascending=[False, False]).to_csv(OUT / "hospital_vent_ascertainment.csv", index=False)

coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
ok = set(h.loc[h.vent_ascertainable, "hospitalid"])
coh["vent_ascertainable"] = coh.hospitalid.isin(ok).astype(int)
coh.to_csv(HERE / "cohort/patient_level_cohort.csv", index=False)

manifest = pd.DataFrame([
    {"file": "terms_drugs.csv", "n_terms": len(a),
     "detail": f"{(a['class']=='vasoactive').sum()} vasoactive support, {(a['class']=='sedative').sum()} sedative"},
    {"file": "terms_urine_sources.csv", "n_terms": len(b), "detail": "urinary sources, quantitative mL only"},
    {"file": "terms_ventilation_crrt.csv", "n_terms": len(c2),
     "detail": "13 invasive, 15 NIV-exclude, 12 vent-setting timing, 2 CRRT"},
    {"file": "hospital_vent_ascertainment.csv", "n_terms": int(h.vent_ascertainable.sum()),
     "detail": f"{int((~h.vent_ascertainable).sum())} hospitals excluded, "
               f"{int(h.loc[~h.vent_ascertainable,'n_patients'].sum())} patients"},
])
manifest["freeze_date"] = FREEZE
manifest.to_csv(OUT / "MANIFEST.csv", index=False)
print(manifest.to_string(index=False))
print(f"\nventilation-ascertainable: {len(ok)} hospitals, "
      f"{coh.vent_ascertainable.sum():,} of {len(coh):,} patients")
print(f"also GCS-eligible: {((coh.vent_ascertainable==1) & coh.patientunitstayid.isin(set(pd.read_csv(HERE/'cohort/patient_level_cohort_eligible.csv').patientunitstayid))).sum():,} patients")
print(f"\nFROZEN {FREEZE} -> {OUT}/")
