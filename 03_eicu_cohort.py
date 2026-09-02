"""
Paper 2, step 3: eICU analysis cohort, Paper 1 filters applied to the frozen
phenotype from 02_eicu_stroke_phenotype.py.

Mirrors 03_code/01_cohort_construction.py filter for filter. Two deviations are
forced by eICU and are logged in the flow table rather than buried:

  - Paper 1 keeps the first ICU stay of the qualifying admission, one stay per
    patient, ordered by intime. eICU de-identification removes real dates, so
    hospital stays cannot be ordered in time; `unitvisitnumber == 1` gives the
    first unit stay of a hospitalization, and for the few patients with more
    than one qualifying hospitalization the lowest patienthealthsystemstayid is
    kept as a deterministic tie-break.
  - Age >89 is reported by eICU as the string "> 89" and is mapped to 90. Paper 1
    caps age at 90 for the same reason, so the two variables are comparable.

Outputs (cohort/):
  cohort_flow_counts.csv      CONSORT-style attrition, same shape as Paper 1's
  patient_level_cohort.csv    one row per patient, with the window scaffold
"""
import pandas as pd
from pathlib import Path

B = "/Volumes/Lexar/research data/eICU dataset/eicu-collaborative-research-database-2.0"
HERE = Path(__file__).parent
OUT = HERE / "cohort"; OUT.mkdir(exist_ok=True)
MIN_LOS_MIN = 720        # 12 h, Paper 1's MIN_ICU_LOS_DAYS = 0.5
HORIZON_MIN = 4320       # 72 h
WINDOW_MIN = 360         # 6 h
N_WINDOWS = 12

ph = pd.read_csv(HERE / "frozen_phenotype/eicu_stroke_stays.csv")
pat = pd.read_csv(f"{B}/patient.csv.gz",
                  usecols=["patientunitstayid", "patienthealthsystemstayid", "uniquepid", "gender",
                           "age", "hospitalid", "wardid", "unittype", "unitadmitsource",
                           "unitvisitnumber", "unitdischargeoffset", "unitdischargestatus",
                           "hospitaldischargeoffset", "hospitaldischargestatus", "hospitaldischargeyear",
                           "apacheadmissiondx", "admissionweight"])
hosp = pd.read_csv(f"{B}/hospital.csv.gz")

d = ph.merge(pat, on="patientunitstayid", how="left", suffixes=("", "_p"))
flow = []
def log(step, d):
    flow.append({"step": step, "n_stays": len(d), "n_patients": d.uniquepid.nunique(),
                 "n_hospitals": d.hospitalid.nunique()})
    print(f"{step:<58} stays {len(d):>6,}  patients {d.uniquepid.nunique():>6,}  hospitals {d.hospitalid.nunique():>4}")

log("1. frozen stroke phenotype (diagnosisstring paths)", d)
d = d[d.excluded_trauma == 0]
log("2. exclude head/face trauma and CNS-trauma paths", d)

d["age_n"] = pd.to_numeric(d.age.replace("> 89", "90"), errors="coerce")
d = d[d.age_n >= 18]
log("3. age >=18 (\"> 89\" mapped to 90)", d)

d = d[d.unitvisitnumber == 1]
log("4. first ICU unit stay of the hospitalization", d)

d = d[d.unitdischargeoffset >= MIN_LOS_MIN]
log("5. ICU LOS >=12 h (>=2 six-hour windows possible)", d)

d = d.sort_values(["uniquepid", "hospitaldischargeyear", "patienthealthsystemstayid"]) \
     .drop_duplicates("uniquepid", keep="first")
log("6. one hospitalization per patient (deviation: see docstring)", d)

# ---- window scaffold, identical to Paper 1's obs_end / max_valid_window ----
d["obs_end_min"] = d.unitdischargeoffset.clip(upper=HORIZON_MIN)
d["max_valid_window"] = (d.obs_end_min // WINDOW_MIN).clip(upper=N_WINDOWS - 1).astype(int)
d["n_windows"] = d.max_valid_window + 1
d["icu_mortality"] = (d.unitdischargestatus == "Expired").astype(int)
d["hospital_mortality"] = (d.hospitaldischargestatus == "Expired").astype(int)
d["icu_los_days"] = d.unitdischargeoffset / 1440
d = d.merge(hosp, on="hospitalid", how="left")

pd.DataFrame(flow).to_csv(OUT / "cohort_flow_counts.csv", index=False)
cols = ["patientunitstayid", "patienthealthsystemstayid", "uniquepid", "hospitalid", "unittype",
        "unitadmitsource", "numbedscategory", "teachingstatus", "region",
        "age_n", "gender", "admissionweight", "stroke_subtype", "unspecified_only", "has_review_path",
        "unitdischargeoffset", "icu_los_days", "obs_end_min", "max_valid_window", "n_windows",
        "icu_mortality", "hospital_mortality", "apacheadmissiondx"]
d[cols].to_csv(OUT / "patient_level_cohort.csv", index=False)

print(f"\n--- cohort summary (MIMIC-IV Paper 1 in brackets) ---")
print(f"patients                 {len(d):,}   [6,368]")
print(f"hospitals                {d.hospitalid.nunique()}   [1]")
print(f"median age               {d.age_n.median():.0f} [{d.age_n.quantile(.25):.0f}-{d.age_n.quantile(.75):.0f}]   [69]")
print(f"female                   {(d.gender=='Female').mean()*100:.1f}%")
print(f"median ICU LOS (days)    {d.icu_los_days.median():.2f}")
print(f"ICU mortality            {d.icu_mortality.mean()*100:.1f}%   [12.0]")
print(f"hospital mortality       {d.hospital_mortality.mean()*100:.1f}%")
print(f"total 6-h windows        {d.n_windows.sum():,}")
print(f"median windows/patient   {d.n_windows.median():.0f} of 12")
print(f"\nsubtype:\n{d.stroke_subtype.value_counts().to_string()}")
print(f"\nunit type:\n{d.unittype.value_counts().head(8).to_string()}")
h = d.groupby("hospitalid").size()
print(f"\nhospitals: median {h.median():.0f} patients (IQR {h.quantile(.25):.0f}-{h.quantile(.75):.0f}, max {h.max()})")
for t in (20, 50, 100):
    print(f"  >={t:>3}: {(h>=t).sum():>3} hospitals covering {h[h>=t].sum():>5,} ({h[h>=t].sum()/h.sum()*100:.0f}%)")
print(f"\nWrote {OUT}/")
