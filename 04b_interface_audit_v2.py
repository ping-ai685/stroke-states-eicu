"""
Paper 2, step 4 (corrected and extended). Supersedes 04_interface_audit.py.

Four changes, all forced by what the first pass showed:

  1. TEMPERATURE was matched on the wrong field. nurseCharting stores it as
     label='Temperature' with valname='Temperature (C)' / 'Temperature (F)';
     the first pass matched the LABEL against "temperature (" and so reported
     0.0%. The real coverage is measured here.

  2. respiratoryCharting was never scanned, yet it is the richest ventilation
     evidence in eICU (respFlowSettings PEEP / Vent Rate / Tidal Volume (set),
     respFlowPtVentData Exhaled TV / Peak Insp. Pressure). respiratoryCare's
     ventstartoffset reaches only 20.5% of stays and 166 of 176 hospitals fall
     below 50% on that interface, so the composite cannot rest on it.

  3. The `medication` table is added as a second source for vasopressor and
     sedative. infusionDrug covers only 42.1% of stays and 122 of 176 hospitals
     are below 50% on it -- relying on infusionDrug alone would delete most of
     the multicentre structure the study exists to examine.

  4. WINDOW-LEVEL coverage is computed for the variables where stay-level
     presence is misleading. physicalExam looked like the best GCS source at
     99.7% of stays, but it carries roughly one row per patient: the score is
     encoded in the path itself (".../GCS/Motor Score/6"), charted about once a
     stay. Presence is not density, and the model needs density.

Outputs (audit/): *_v2.csv equivalents, plus window_coverage_by_variable.csv and
hospital_eligibility.csv.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

B = "/Volumes/Lexar/research data/eICU dataset/eicu-collaborative-research-database-2.0"
HERE = Path(__file__).parent
OUT = HERE / "audit"; OUT.mkdir(exist_ok=True)
WINDOW, NW = 360, 12

coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv",
                  usecols=["patientunitstayid", "hospitalid", "unittype", "max_valid_window", "n_windows"])
STAYS = set(coh.patientunitstayid)
MAXW = coh.set_index("patientunitstayid").max_valid_window
TOTAL_WINDOWS = int(coh.n_windows.sum())
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
log(f"start -- {len(STAYS):,} stays, {TOTAL_WINDOWS:,} windows")

iface, var, wins = {}, {}, {}
def add_stay(d, k, ids): d.setdefault(k, set()).update(ids & STAYS)
def add_win(k, df, off):
    """record (stay, window) pairs for rows whose offset falls in windows 0..11"""
    w = df[off] // WINDOW
    ok = (w >= 0) & (w <= df.patientunitstayid.map(MAXW))
    wins.setdefault(k, set()).update(zip(df.loc[ok, "patientunitstayid"], w[ok].astype(int)))

def scan(table, usecols, fn, chunk=2_000_000):
    t0 = time.time(); n = 0
    for ch in pd.read_csv(f"{B}/{table}.csv.gz", usecols=usecols, chunksize=chunk, low_memory=False):
        ch = ch[ch.patientunitstayid.isin(STAYS)]
        if len(ch): fn(ch)
        n += 1
    log(f"  {table}: {n} chunks, {time.time()-t0:.0f}s")

# ---- small tables ----
def f_diag(ch): add_stay(iface, "diagnosis", set(ch.patientunitstayid))
scan("diagnosis", ["patientunitstayid"], f_diag, 4_000_000)

def f_treat(ch):
    add_stay(iface, "treatment", set(ch.patientunitstayid))
    s = ch.treatmentstring.fillna("").str.lower()
    add_stay(var, "crrt__treatment", set(ch.loc[s.str.startswith("renal|dialysis"), "patientunitstayid"]))
    add_stay(var, "mech_vent__treatment",
             set(ch.loc[s.str.contains("mechanical ventilation"), "patientunitstayid"]))
scan("treatment", ["patientunitstayid", "treatmentstring"], f_treat, 4_000_000)

def f_rc(ch):
    add_stay(iface, "respiratoryCare", set(ch.patientunitstayid))
    add_stay(var, "mech_vent__respCare",
             set(ch.loc[ch.ventstartoffset.notna() & (ch.ventstartoffset != 0), "patientunitstayid"]))
scan("respiratoryCare", ["patientunitstayid", "ventstartoffset"], f_rc, 4_000_000)

PRESS = "norepinephrine|levophed|epinephrine|phenylephrine|neo-synephrine|dopamine|dobutamine|vasopressin|milrinone"
SED   = "propofol|diprivan|midazolam|versed|dexmedetomidine|precedex|lorazepam|ativan"
def f_inf(ch):
    add_stay(iface, "infusionDrug", set(ch.patientunitstayid))
    d = ch.drugname.fillna("").str.lower()
    for k, rx in [("vasopressor__infusionDrug", PRESS), ("sedative__infusionDrug", SED)]:
        m = d.str.contains(rx, regex=True)
        add_stay(var, k, set(ch.loc[m, "patientunitstayid"]))
        add_win(k, ch.loc[m], "infusionoffset")
scan("infusionDrug", ["patientunitstayid", "infusionoffset", "drugname"], f_inf, 4_000_000)

def f_med(ch):
    add_stay(iface, "medication", set(ch.patientunitstayid))
    d = ch.drugname.fillna("").str.lower()
    iv = ch.routeadmin.fillna("").str.lower().str.contains("iv|intraven", regex=True)
    for k, rx in [("vasopressor__medication", PRESS), ("sedative__medication", SED)]:
        add_stay(var, k, set(ch.loc[d.str.contains(rx, regex=True) & iv, "patientunitstayid"]))
scan("medication", ["patientunitstayid", "drugname", "routeadmin"], f_med, 4_000_000)

def f_pe(ch):
    add_stay(iface, "physicalExam", set(ch.patientunitstayid))
    p = ch.physicalexampath.fillna("")
    for k, pat in [("gcs_eye__physicalExam", "GCS/Eyes Score/"), ("gcs_motor__physicalExam", "GCS/Motor Score/")]:
        m = p.str.contains(pat, regex=False)
        add_stay(var, k, set(ch.loc[m, "patientunitstayid"]))
        add_win(k, ch.loc[m], "physicalexamoffset")
scan("physicalExam", ["patientunitstayid", "physicalexamoffset", "physicalexampath"], f_pe, 4_000_000)

def f_io(ch):
    add_stay(iface, "intakeOutput", set(ch.patientunitstayid))
    lab = ch.celllabel.fillna("").str.lower(); path = ch.cellpath.fillna("").str.lower()
    m = path.str.contains("output") & lab.str.contains("urine|foley|void|catheter", regex=True)
    add_stay(var, "urine_output__intakeOutput", set(ch.loc[m, "patientunitstayid"]))
    add_win("urine_output", ch.loc[m], "intakeoutputoffset")
scan("intakeOutput", ["patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel"], f_io, 4_000_000)

LABS = {"wbc": "wbc x 1000", "hemoglobin": "hgb", "platelet": "platelets x 1000",
        "creatinine": "creatinine", "bun": "bun", "sodium": "sodium",
        "potassium": "potassium", "glucose": "glucose", "glucose_bedside": "bedside glucose"}
def f_lab(ch):
    add_stay(iface, "lab", set(ch.patientunitstayid))
    nm = ch.labname.fillna("").str.lower()
    for v, key in LABS.items():
        m = nm == key
        add_stay(var, f"{v}__lab", set(ch.loc[m, "patientunitstayid"]))
        add_win(v, ch.loc[m], "labresultoffset")
scan("lab", ["patientunitstayid", "labresultoffset", "labname"], f_lab, 4_000_000)

def f_rch(ch):
    add_stay(iface, "respiratoryCharting", set(ch.patientunitstayid))
    lbl = ch.respchartvaluelabel.fillna("")
    vent = lbl.isin(["PEEP", "Vent Rate", "Tidal Volume (set)", "Exhaled TV (patient)",
                     "Peak Insp. Pressure", "Mean Airway Pressure", "Exhaled MV",
                     "Pressure Support", "TV/kg IBW", "RR (patient)", "Total RR"])
    add_stay(var, "mech_vent__respCharting", set(ch.loc[vent, "patientunitstayid"]))
    add_win("mech_vent__respCharting", ch.loc[vent], "respchartoffset")
scan("respiratoryCharting", ["patientunitstayid", "respchartoffset", "respcharttypecat",
                             "respchartvaluelabel"], f_rch)

def f_va(ch):
    add_stay(iface, "vitalAperiodic", set(ch.patientunitstayid))
    for v, c in [("sbp__cuff", "noninvasivesystolic"), ("map__cuff", "noninvasivemean")]:
        m = ch[c].notna()
        add_stay(var, v, set(ch.loc[m, "patientunitstayid"])); add_win(v, ch.loc[m], "observationoffset")
scan("vitalAperiodic", ["patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivemean"], f_va, 4_000_000)

def f_vp(ch):
    add_stay(iface, "vitalPeriodic", set(ch.patientunitstayid))
    for v, c in [("heart_rate", "heartrate"), ("resp_rate", "respiration"), ("spo2", "sao2"),
                 ("temp_c__vitalPeriodic", "temperature"), ("sbp__arterial", "systemicsystolic"),
                 ("map__arterial", "systemicmean")]:
        m = ch[c].notna()
        add_stay(var, v, set(ch.loc[m, "patientunitstayid"])); add_win(v, ch.loc[m], "observationoffset")
scan("vitalPeriodic", ["patientunitstayid", "observationoffset", "heartrate", "respiration", "sao2",
                       "temperature", "systemicsystolic", "systemicmean"], f_vp)

def f_nc(ch):
    add_stay(iface, "nurseCharting", set(ch.patientunitstayid))
    lbl = ch.nursingchartcelltypevallabel.fillna("")
    nm = ch.nursingchartcelltypevalname.fillna("")
    gcs = lbl == "Glasgow coma score"
    for k, sub in [("gcs_eye__nurseCharting", "Eyes"), ("gcs_motor__nurseCharting", "Motor")]:
        m = gcs & (nm == sub)
        add_stay(var, k, set(ch.loc[m, "patientunitstayid"])); add_win(k, ch.loc[m], "nursingchartoffset")
    # CORRECTED: label is 'Temperature', the unit lives in valname
    t = (lbl == "Temperature") & nm.isin(["Temperature (C)", "Temperature (F)"])
    add_stay(var, "temp_c__nurseCharting", set(ch.loc[t, "patientunitstayid"]))
    add_win("temp_c__nurseCharting", ch.loc[t], "nursingchartoffset")
scan("nurseCharting", ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel",
                       "nursingchartcelltypevalname"], f_nc)

# ---------------------------------------------------------------------------
def frame(d, name):
    f = pd.DataFrame({"patientunitstayid": sorted(STAYS)})
    for k in sorted(d): f[k] = f.patientunitstayid.isin(d[k]).astype(int)
    f.to_csv(OUT / name, index=False); return f
pi = frame(iface, "interface_presence_by_stay_v2.csv")
pv = frame(var, "variable_presence_by_stay_v2.csv")

wc = pd.DataFrame([{"variable": k, "windows_covered": len(v),
                    "pct_of_all_windows": round(100 * len(v) / TOTAL_WINDOWS, 1)}
                   for k, v in sorted(wins.items())]).sort_values("pct_of_all_windows", ascending=False)
wc.to_csv(OUT / "window_coverage_by_variable.csv", index=False)

g = pi.merge(coh, on="patientunitstayid").groupby("hospitalid")
cov = g[[c for c in pi.columns if c != "patientunitstayid"]].mean()
cov.insert(0, "n_patients", g.size())
cov.round(3).to_csv(OUT / "interface_coverage_by_hospital_v2.csv")

print(f"\n=== interface coverage ({len(STAYS):,} stays) ===")
for c in sorted(iface): print(f"  {c:<22} {pi[c].mean()*100:5.1f}%   hospitals <50%: {(cov[c]<0.5).sum():>3}/{len(cov)}")
print(f"\n=== window-level coverage (of {TOTAL_WINDOWS:,} windows, before forward fill) ===")
print(wc.to_string(index=False))
print(f"\n=== composite organ-support reach (stay level) ===")
for name, cols in [("mech_vent", ["mech_vent__respCare", "mech_vent__respCharting", "mech_vent__treatment"]),
                   ("vasopressor", ["vasopressor__infusionDrug", "vasopressor__medication"]),
                   ("sedative", ["sedative__infusionDrug", "sedative__medication"])]:
    have = [c for c in cols if c in pv.columns]
    print(f"  {name:<12} any source {pv[have].max(axis=1).mean()*100:5.1f}%   " +
          "  ".join(f"{c.split('__')[1]} {pv[c].mean()*100:.1f}%" for c in have))
log("done")
