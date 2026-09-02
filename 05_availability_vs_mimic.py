"""
Paper 2, step 5: the number that decides hospital eligibility -- eICU window-level
data availability measured against MIMIC-IV on the same definition.

Stay-level presence (step 4) is not the right yardstick. What the HMM consumes is
a 6-hour window with a value in it, so the comparison has to be
"share of windows with a measured value, before any forward fill", computed
identically in both databases. MIMIC's side comes from the `{var}_missing_raw`
indicators that 03_impute_missing.py wrote for exactly this purpose.

Also settles two questions step 4 left open:
  - hospital eligibility tiers, driven by GCS window coverage rather than by
    patient count alone
  - whether eICU can separate INVASIVE ventilation from non-invasive, which is
    what Paper 1's mech_vent flag means (procedureevents 225792)

Outputs (audit/):
  availability_vs_mimic.csv
  hospital_eligibility.csv
  gcs_coverage_by_stay.csv
  ventilation_source_separability.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

B = "/Volumes/Lexar/research data/eICU dataset/eicu-collaborative-research-database-2.0"
HERE = Path(__file__).parent
T1 = HERE.parent / "04_outputs/tables"
OUT = HERE / "audit"

VARS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c", "gcs_eye", "gcs_motor",
        "urine_output_ml", "wbc", "hemoglobin", "platelet", "creatinine", "bun",
        "sodium", "potassium", "glucose"]
EMAP = {"sbp": "sbp__cuff", "map": "map__cuff", "temp_c": "temp_c__nurseCharting",
        "gcs_eye": "gcs_eye__nurseCharting", "gcs_motor": "gcs_motor__nurseCharting",
        "urine_output_ml": "urine_output"}

# ---- MIMIC side ----
m = pd.read_csv(T1 / "timewindow_level_imputed.csv")
mim = {v: (1 - m[f"{v}_missing_raw"].mean()) * 100 for v in VARS}
binp = {b: m[b].mean() * 100 for b in ["mech_vent", "crrt", "vasopressor", "sedative"]}

# ---- eICU side ----
e = pd.read_csv(OUT / "window_coverage_by_variable.csv").set_index("variable").pct_of_all_windows
rows = []
for v in VARS:
    a, b = mim[v], float(e.get(EMAP.get(v, v), np.nan))
    rows.append({"variable": v, "mimic_pct_windows": round(a, 1), "eicu_pct_windows": round(b, 1),
                 "difference": round(b - a, 1),
                 "verdict": "equivalent" if b - a > -10 else ("watch" if b - a > -25 else "PROBLEM")})
av = pd.DataFrame(rows)
av.to_csv(OUT / "availability_vs_mimic.csv", index=False)
print(av.to_string(index=False))
print("\nMIMIC window-level prevalence of the binary supports (the transport target):")
for k, v in binp.items(): print(f"  {k:<14}{v:>6.1f}%")

# ---- GCS coverage per stay, then hospital eligibility ----
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
STAYS = set(coh.patientunitstayid); MAXW = coh.set_index("patientunitstayid").max_valid_window
hit = {}
for ch in pd.read_csv(f"{B}/nurseCharting.csv.gz",
                      usecols=["patientunitstayid", "nursingchartoffset",
                               "nursingchartcelltypevallabel", "nursingchartcelltypevalname"],
                      chunksize=2_000_000, low_memory=False):
    ch = ch[ch.patientunitstayid.isin(STAYS)]
    ch = ch[(ch.nursingchartcelltypevallabel.fillna("") == "Glasgow coma score") &
            (ch.nursingchartcelltypevalname.fillna("").isin(["Eyes", "Motor"]))]
    if len(ch):
        w = ch.nursingchartoffset // 360
        ok = (w >= 0) & (w <= ch.patientunitstayid.map(MAXW))
        for sid, ww in zip(ch.loc[ok, "patientunitstayid"], w[ok].astype(int)):
            hit.setdefault(sid, set()).add(ww)
coh["gcs_windows"] = coh.patientunitstayid.map(lambda s: len(hit.get(s, ())))
coh["gcs_frac"] = coh.gcs_windows / coh.n_windows
coh[["patientunitstayid", "hospitalid", "n_windows", "gcs_windows", "gcs_frac"]] \
    .to_csv(OUT / "gcs_coverage_by_stay.csv", index=False)
print(f"\nPatients with NO GCS in any window: {(coh.gcs_frac == 0).sum():,} ({(coh.gcs_frac == 0).mean()*100:.1f}%)")

h = coh.groupby("hospitalid").agg(n=("patientunitstayid", "size"), gcs=("gcs_frac", "mean"))
print(f"Hospitals charting no GCS at all: {(h.gcs == 0).sum()} of {len(h)}; below 25%: {(h.gcs < 0.25).sum()}")
tiers = []
for lbl, hs in [("n>=20", h[h.n >= 20]), ("n>=50", h[h.n >= 50]), ("n>=100", h[h.n >= 100]),
                ("n>=20 & GCS>=25%", h[(h.n >= 20) & (h.gcs >= .25)]),
                ("n>=20 & GCS>=50%", h[(h.n >= 20) & (h.gcs >= .50)]),
                ("n>=50 & GCS>=50%", h[(h.n >= 50) & (h.gcs >= .50)]),
                ("n>=100 & GCS>=50%", h[(h.n >= 100) & (h.gcs >= .50)])]:
    p = coh[coh.hospitalid.isin(hs.index)]
    tiers.append({"rule": lbl, "hospitals": len(hs), "patients": len(p),
                  "pct_of_cohort": round(len(p) / len(coh) * 100)})
tiers = pd.DataFrame(tiers); tiers.to_csv(OUT / "hospital_eligibility.csv", index=False)
print("\n" + tiers.to_string(index=False))

# ---- can eICU separate invasive from non-invasive ventilation? ----
t = pd.read_csv(f"{B}/treatment.csv.gz", usecols=["patientunitstayid", "treatmentstring"])
t = t[t.patientunitstayid.isin(STAYS)]
rc = pd.read_csv(f"{B}/respiratoryCare.csv.gz", usecols=["patientunitstayid", "airwaytype"], low_memory=False)
rc = rc[rc.patientunitstayid.isin(STAYS)]
ts = t.treatmentstring.fillna("")
inv = set(t.loc[ts.str.startswith("pulmonary|ventilation and oxygenation|mechanical ventilation"), "patientunitstayid"])
niv = set(t.loc[ts.str.contains("non-invasive ventilation|CPAP/PEEP therapy", regex=True), "patientunitstayid"])
aw  = set(rc.loc[rc.airwaytype.isin(["Oral ETT", "Nasal ETT", "Tracheostomy"]), "patientunitstayid"])
vs = pd.DataFrame([
    {"source": "treatment: |mechanical ventilation% (INVASIVE, definite)", "stays": len(inv)},
    {"source": "treatment: non-invasive ventilation or CPAP/PEEP (must be EXCLUDED)", "stays": len(niv)},
    {"source": "respiratoryCare airwaytype ETT/tracheostomy (INVASIVE, definite)", "stays": len(aw)},
    {"source": "either definite-invasive source", "stays": len(inv | aw)},
    {"source": "respiratoryCare rows with airwaytype populated (%)",
     "stays": round(rc.airwaytype.notna().mean() * 100, 1)},
])
vs.to_csv(OUT / "ventilation_source_separability.csv", index=False)
print("\n" + vs.to_string(index=False))
print(f"\ncohort size {len(coh):,}; definite-invasive reach {len(inv|aw)/len(coh)*100:.1f}% of stays")
