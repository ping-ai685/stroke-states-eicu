"""
Paper 2, step 7: build the eICU 6-hour window table.

Scope: the 17 CONTINUOUS variables only. The four organ-support flags are
deferred until their term lists (drug names, urine source labels, ventilation
evidence) have been clinically reviewed and frozen -- checklist item 5. This is
not a compromise: the 17 continuous variables are exactly the feature set of the
treatment-free specificity control (protocol v1.1 section 10.3), so the control
can be decoded as soon as this table exists, without waiting on any term list.

Every rule is the discovery pipeline's, applied to eICU sources:
  - window_idx = floor(offset / 360), kept while 0 <= idx <= max_valid_window
  - continuous variables: LAST value in the window
  - urine output: SUM over the window
  - arterial BP preferred over cuff, coalesced AFTER windowing (never row-wise)
  - temperature: vitalPeriodic (Celsius) preferred, then nurseCharting with
    Fahrenheit converted
  - GCS: numeric cast; non-numeric strings such as "Unable to score due to
    medication" become MISSING, never floor-coded to 1
  - physiologically impossible values -> missing, using the discovery bounds

Imputation and standardization are NOT done here; they are step 10, so the raw
table stays inspectable exactly as `timewindow_level_raw.csv` does in Paper 1.

Outputs (windows/):
  timewindow_level_raw_eicu.csv
  extraction_qa.csv
"""
import numpy as np
import pandas as pd
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import data_paths                                    # noqa: E402

B = str(data_paths.eicu())
HERE = Path(__file__).parent
OUT = HERE / "windows"; OUT.mkdir(exist_ok=True)
WINDOW, NW = 360, 12

IMPOSSIBLE = {"heart_rate": (20, 300), "sbp": (40, 300), "map": (20, 250), "resp_rate": (0, 80),
              "spo2": (0, 100), "temp_c": (25, 45), "gcs_eye": (1, 4), "gcs_motor": (1, 6),
              "urine_output_ml": (0, 20000), "wbc": (0, 100), "hemoglobin": (2, 20),
              "platelet": (0, 2000), "creatinine": (0, 20), "bun": (0, 200),
              "sodium": (100, 200), "potassium": (1, 10), "glucose": (0, 1500)}

coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv",
                  usecols=["patientunitstayid", "hospitalid", "max_valid_window", "n_windows"])
STAYS = set(coh.patientunitstayid)
MAXW = coh.set_index("patientunitstayid").max_valid_window
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
log(f"start -- {len(STAYS):,} stays, {int(coh.n_windows.sum()):,} windows")

def windowed(df, offcol):
    """attach window_idx and drop rows outside the patient's observed windows"""
    w = df[offcol] // WINDOW
    keep = (w >= 0) & (w <= df.patientunitstayid.map(MAXW))
    out = df.loc[keep].copy()
    out["window_idx"] = w[keep].astype(int)
    return out

def reduce_last(frames, name):
    """last value per (stay, window), resolving across chunks by offset then rank"""
    if not frames:
        return pd.Series(name=name, dtype="float64")
    d = pd.concat(frames, ignore_index=True)
    d = d.sort_values(["patientunitstayid", "window_idx", "_rank", "_off"])
    d = d.drop_duplicates(["patientunitstayid", "window_idx"], keep="last")
    return d.set_index(["patientunitstayid", "window_idx"])["_val"].rename(name)

acc = {}
def push(name, df, offcol, valcol, rank=0):
    d = windowed(df[df[valcol].notna()], offcol)
    if not len(d):
        return
    d = d.rename(columns={offcol: "_off", valcol: "_val"})
    d["_rank"] = rank
    d = d.sort_values(["patientunitstayid", "window_idx", "_rank", "_off"]) \
         .drop_duplicates(["patientunitstayid", "window_idx"], keep="last")
    acc.setdefault(name, []).append(d[["patientunitstayid", "window_idx", "_off", "_rank", "_val"]])

def scan(table, usecols, fn, chunk=2_000_000):
    t0 = time.time(); n = 0
    for ch in pd.read_csv(f"{B}/{table}.csv.gz", usecols=usecols, chunksize=chunk, low_memory=False):
        ch = ch[ch.patientunitstayid.isin(STAYS)]
        if len(ch): fn(ch)
        n += 1
    log(f"  {table}: {n} chunks, {time.time()-t0:.0f}s")

# ---- labs -----------------------------------------------------------------
LABS = {"wbc": "wbc x 1000", "hemoglobin": "hgb", "platelet": "platelets x 1000",
        "creatinine": "creatinine", "bun": "bun", "sodium": "sodium",
        "potassium": "potassium", "glucose": "glucose"}
def f_lab(ch):
    nm = ch.labname.fillna("").str.lower()
    for v, key in LABS.items():
        sub = ch[nm == key]
        if len(sub): push(v, sub, "labresultoffset", "labresult")
scan("lab", ["patientunitstayid", "labresultoffset", "labname", "labresult"], f_lab, 4_000_000)

# ---- non-invasive BP (rank 0 = fallback) ----------------------------------
def f_va(ch):
    push("sbp", ch, "observationoffset", "noninvasivesystolic", rank=0)
    push("map", ch, "observationoffset", "noninvasivemean", rank=0)
scan("vitalAperiodic", ["patientunitstayid", "observationoffset",
                        "noninvasivesystolic", "noninvasivemean"], f_va, 4_000_000)

# ---- periodic vitals; arterial BP is rank 1 so it wins the coalesce --------
def f_vp(ch):
    push("heart_rate", ch, "observationoffset", "heartrate")
    push("resp_rate", ch, "observationoffset", "respiration")
    push("spo2", ch, "observationoffset", "sao2")
    push("temp_c", ch, "observationoffset", "temperature", rank=1)
    push("sbp", ch, "observationoffset", "systemicsystolic", rank=1)
    push("map", ch, "observationoffset", "systemicmean", rank=1)
scan("vitalPeriodic", ["patientunitstayid", "observationoffset", "heartrate", "respiration",
                       "sao2", "temperature", "systemicsystolic", "systemicmean"], f_vp)

# ---- GCS and fallback temperature from nurseCharting ----------------------
gcs_nonnumeric = {"eye": 0, "motor": 0}
def f_nc(ch):
    lbl = ch.nursingchartcelltypevallabel.fillna("")
    nm = ch.nursingchartcelltypevalname.fillna("")
    gcs = lbl == "Glasgow coma score"
    for var, sub in [("gcs_eye", "Eyes"), ("gcs_motor", "Motor")]:
        d = ch[gcs & (nm == sub)].copy()
        if not len(d): continue
        d["_num"] = pd.to_numeric(d.nursingchartvalue, errors="coerce")
        gcs_nonnumeric[var.split("_")[1]] += int(d._num.isna().sum())
        push(var, d, "nursingchartoffset", "_num")
    t = ch[(lbl == "Temperature") & nm.isin(["Temperature (C)", "Temperature (F)"])].copy()
    if len(t):
        t["_num"] = pd.to_numeric(t.nursingchartvalue, errors="coerce")
        f = nm.loc[t.index] == "Temperature (F)"
        t.loc[f, "_num"] = (t.loc[f, "_num"] - 32) * 5 / 9
        push("temp_c", t, "nursingchartoffset", "_num", rank=0)   # loses to vitalPeriodic
scan("nurseCharting", ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel",
                       "nursingchartcelltypevalname", "nursingchartvalue"], f_nc)

# ---- urine output: SUM per window -----------------------------------------
urine_parts = []
def f_io(ch):
    lab = ch.celllabel.fillna("").str.lower(); path = ch.cellpath.fillna("").str.lower()
    d = ch[path.str.contains("output") & lab.str.contains("urine|foley|void|catheter", regex=True)]
    d = windowed(d[d.cellvaluenumeric.notna()], "intakeoutputoffset")
    if len(d):
        urine_parts.append(d.groupby(["patientunitstayid", "window_idx"])["cellvaluenumeric"].sum())
scan("intakeOutput", ["patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel",
                      "cellvaluenumeric"], f_io, 4_000_000)

# ---- assemble --------------------------------------------------------------
log("assembling ...")
rows = [(s, w) for s in sorted(STAYS) for w in range(int(MAXW[s]) + 1)]
wide = pd.DataFrame(rows, columns=["patientunitstayid", "window_idx"])
for name in ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c", "gcs_eye", "gcs_motor",
             "wbc", "hemoglobin", "platelet", "creatinine", "bun", "sodium", "potassium", "glucose"]:
    s = reduce_last(acc.get(name, []), name)
    wide = wide.merge(s, left_on=["patientunitstayid", "window_idx"], right_index=True, how="left")
u = pd.concat(urine_parts).groupby(level=[0, 1]).sum().rename("urine_output_ml") if urine_parts else None
wide = wide.merge(u, left_on=["patientunitstayid", "window_idx"], right_index=True, how="left")

# ---- physiological range check, discovery bounds ---------------------------
qa = []
for v, (lo, hi) in IMPOSSIBLE.items():
    n_before = int(wide[v].notna().sum())
    bad = wide[v].notna() & ((wide[v] < lo) | (wide[v] > hi))
    wide.loc[bad, v] = np.nan
    qa.append({"variable": v, "measured_windows": n_before, "set_missing_out_of_range": int(bad.sum()),
               "pct_windows_measured": round(wide[v].notna().mean() * 100, 1),
               "median": round(float(wide[v].median()), 2) if wide[v].notna().any() else None})
qa = pd.DataFrame(qa)
wide.to_csv(OUT / "timewindow_level_raw_eicu.csv", index=False)
qa.to_csv(OUT / "extraction_qa.csv", index=False)

log("done")
print(f"\nwindows built: {len(wide):,} (expected {int(coh.n_windows.sum()):,})")
print(f"GCS rows with a non-numeric value, treated as missing: "
      f"eye {gcs_nonnumeric['eye']:,}, motor {gcs_nonnumeric['motor']:,}")
print("\n" + qa.to_string(index=False))
