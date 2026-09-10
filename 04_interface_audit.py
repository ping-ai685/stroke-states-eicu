"""
Paper 2, step 4: blind hospital/unit data-interface availability audit.

Protocol v1.0 EN section 8.1. Run BEFORE any state assignment and using no
outcome information: this script never reads a mortality field and never touches
the frozen model. Its only job is to establish, per hospital, which data
interfaces are actually connected -- eICU's documentation is explicit that a
variable absent at a hospital usually means the interface was never wired up,
not that the measurement was not taken.

Amendment A3 adds one interface that the protocol did not anticipate: whether a
hospital populates the `diagnosis` table at all. 2,006 stays that
apacheadmissiondx calls stroke carry no stroke diagnosisstring, 18% of them have
no diagnosis row whatsoever, and the misses concentrate in a handful of
hospitals -- so diagnosis-table coverage is itself an eligibility criterion.

Restricted to the cohort from 03_eicu_cohort.py. Writes incrementally, one CSV
per source table, so a long scan that dies partway does not lose everything.

Outputs (audit/):
  interface_presence_by_stay.csv    one row per cohort stay, one column per interface
  interface_coverage_by_hospital.csv  hospital x interface, share of stays with any data
  variable_presence_by_stay.csv     one row per cohort stay, one column per model variable
  audit_progress.log                which tables have completed
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys, time

import sys
sys.path.insert(0, str(Path(__file__).parent))
import data_paths                                    # noqa: E402

B = str(data_paths.eicu())
HERE = Path(__file__).parent
OUT = HERE / "audit"; OUT.mkdir(exist_ok=True)
LOG = OUT / "audit_progress.log"
CHUNK = 4_000_000

cohort = pd.read_csv(HERE / "cohort/patient_level_cohort.csv",
                     usecols=["patientunitstayid", "hospitalid", "unittype"])
STAYS = set(cohort.patientunitstayid)
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")
log(f"audit start -- {len(STAYS):,} cohort stays, {cohort.hospitalid.nunique()} hospitals")

# accumulators: interface -> set of stays with >=1 row; variable -> set of stays
iface, var = {}, {}
def mark(d, key, ids):
    d.setdefault(key, set()).update(ids & STAYS)

def scan(table, usecols, handler, chunk=CHUNK):
    t0 = time.time(); n = 0
    for ch in pd.read_csv(f"{B}/{table}.csv.gz", usecols=usecols, chunksize=chunk, low_memory=False):
        ch = ch[ch.patientunitstayid.isin(STAYS)]
        if len(ch): handler(ch)
        n += 1
    log(f"  {table}: {n} chunks in {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# small tables first, so failures surface fast
# ---------------------------------------------------------------------------
def h_diag(ch): mark(iface, "diagnosis", set(ch.patientunitstayid))
scan("diagnosis", ["patientunitstayid"], h_diag)

def h_treat(ch):
    mark(iface, "treatment", set(ch.patientunitstayid))
    s = ch.treatmentstring.fillna("").str.lower()
    mark(var, "crrt__treatment", set(ch.loc[s.str.startswith("renal|dialysis"), "patientunitstayid"]))
    mark(var, "mech_vent__treatment",
         set(ch.loc[s.str.contains("ventilation and oxygenation|mechanical ventilation", regex=True),
                    "patientunitstayid"]))
scan("treatment", ["patientunitstayid", "treatmentstring"], h_treat)

def h_rc(ch):
    mark(iface, "respiratoryCare", set(ch.patientunitstayid))
    mark(var, "mech_vent__respCare_ventoffset",
         set(ch.loc[ch.ventstartoffset.notna() & (ch.ventstartoffset != 0), "patientunitstayid"]))
scan("respiratoryCare", ["patientunitstayid", "ventstartoffset"], h_rc)

def h_infu(ch):
    mark(iface, "infusionDrug", set(ch.patientunitstayid))
    d = ch.drugname.fillna("").str.lower()
    press = d.str.contains("norepinephrine|levophed|epinephrine|phenylephrine|neo-synephrine|"
                           "dopamine|dobutamine|vasopressin|milrinone", regex=True)
    sed = d.str.contains("propofol|diprivan|midazolam|versed|dexmedetomidine|precedex|lorazepam|ativan",
                         regex=True)
    mark(var, "vasopressor__infusionDrug", set(ch.loc[press, "patientunitstayid"]))
    mark(var, "sedative__infusionDrug", set(ch.loc[sed, "patientunitstayid"]))
scan("infusionDrug", ["patientunitstayid", "drugname"], h_infu)

def h_pe(ch):
    mark(iface, "physicalExam", set(ch.patientunitstayid))
    mark(var, "gcs__physicalExam",
         set(ch.loc[ch.physicalexampath.fillna("").str.contains("GCS", case=False), "patientunitstayid"]))
scan("physicalExam", ["patientunitstayid", "physicalexampath"], h_pe)

def h_io(ch):
    mark(iface, "intakeOutput", set(ch.patientunitstayid))
    lab = ch.celllabel.fillna("").str.lower(); path = ch.cellpath.fillna("").str.lower()
    urine = path.str.contains("output") & lab.str.contains("urine|foley|void|catheter", regex=True)
    mark(var, "urine_output__intakeOutput", set(ch.loc[urine, "patientunitstayid"]))
scan("intakeOutput", ["patientunitstayid", "cellpath", "celllabel"], h_io)

LABS = {"wbc": "wbc x 1000", "hemoglobin": "hgb", "platelet": "platelets x 1000",
        "creatinine": "creatinine", "bun": "bun", "sodium": "sodium",
        "potassium": "potassium", "glucose": "glucose", "glucose_bedside": "bedside glucose"}
def h_lab(ch):
    mark(iface, "lab", set(ch.patientunitstayid))
    nm = ch.labname.fillna("").str.lower()
    for v, key in LABS.items():
        mark(var, f"{v}__lab", set(ch.loc[nm == key, "patientunitstayid"]))
scan("lab", ["patientunitstayid", "labname"], h_lab)

def h_va(ch):
    mark(iface, "vitalAperiodic", set(ch.patientunitstayid))
    for v, c in [("sbp__cuff", "noninvasivesystolic"), ("map__cuff", "noninvasivemean")]:
        mark(var, v, set(ch.loc[ch[c].notna(), "patientunitstayid"]))
scan("vitalAperiodic", ["patientunitstayid", "noninvasivesystolic", "noninvasivemean"], h_va)

def h_vp(ch):
    mark(iface, "vitalPeriodic", set(ch.patientunitstayid))
    for v, c in [("heart_rate__vitalPeriodic", "heartrate"), ("resp_rate__vitalPeriodic", "respiration"),
                 ("spo2__vitalPeriodic", "sao2"), ("temp_c__vitalPeriodic", "temperature"),
                 ("sbp__arterial", "systemicsystolic"), ("map__arterial", "systemicmean")]:
        mark(var, v, set(ch.loc[ch[c].notna(), "patientunitstayid"]))
scan("vitalPeriodic", ["patientunitstayid", "heartrate", "respiration", "sao2", "temperature",
                       "systemicsystolic", "systemicmean"], h_vp, chunk=2_000_000)

def h_nc(ch):
    mark(iface, "nurseCharting", set(ch.patientunitstayid))
    lbl = ch.nursingchartcelltypevallabel.fillna("")
    gcs = lbl.str.lower() == "glasgow coma score"
    nm = ch.nursingchartcelltypevalname.fillna("")
    mark(var, "gcs_eye__nurseCharting", set(ch.loc[gcs & (nm == "Eyes"), "patientunitstayid"]))
    mark(var, "gcs_motor__nurseCharting", set(ch.loc[gcs & (nm == "Motor"), "patientunitstayid"]))
    mark(var, "temp_c__nurseCharting",
         set(ch.loc[lbl.str.lower().str.startswith("temperature ("), "patientunitstayid"]))
scan("nurseCharting", ["patientunitstayid", "nursingchartcelltypevallabel",
                       "nursingchartcelltypevalname"], h_nc, chunk=2_000_000)

# ---------------------------------------------------------------------------
def to_frame(d, name):
    f = pd.DataFrame({"patientunitstayid": sorted(STAYS)})
    for k in sorted(d):
        f[k] = f.patientunitstayid.isin(d[k]).astype(int)
    f.to_csv(OUT / name, index=False)
    return f

pres_i = to_frame(iface, "interface_presence_by_stay.csv")
pres_v = to_frame(var, "variable_presence_by_stay.csv")
cov = pres_i.merge(cohort, on="patientunitstayid").groupby("hospitalid")
rows = cov[[c for c in pres_i.columns if c != "patientunitstayid"]].mean()
rows.insert(0, "n_patients", cov.size())
rows.round(3).to_csv(OUT / "interface_coverage_by_hospital.csv")

log("audit complete")
print("\n=== interface coverage across the cohort ===")
for c in sorted(iface): print(f"  {c:<22} {pres_i[c].mean()*100:5.1f}% of stays")
print("\n=== variable presence across the cohort ===")
for c in sorted(var): print(f"  {c:<34} {pres_v[c].mean()*100:5.1f}% of stays")
print(f"\nHospitals with <50% coverage, by interface:")
for c in sorted(iface):
    n = (rows[c] < 0.5).sum()
    print(f"  {c:<22} {n:>3} of {len(rows)} hospitals")
