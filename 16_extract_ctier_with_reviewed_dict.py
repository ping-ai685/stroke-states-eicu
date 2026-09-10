"""
Paper 2, checklist item 5c: technical check of the reviewed dictionaries.

The reviewer asked for exactly this before freezing: re-run the eICU extraction
with the adjudicated term lists and look ONLY at capture prevalence, hospital
coverage and unrecognised new terms. No outcome field and no state assignment is
touched here.

Reconstruction rules, all pre-specified in protocol v1.1 section 7.3:

  mech_vent   invasive only. Definite-invasive events are treatment rows the
              reviewer marked INVASIVE vent, plus respiratoryCare airway rows for
              ETT/tracheostomy. respiratoryCharting vent-setting labels give
              TIMING only, and are used only in stays that have definite-invasive
              evidence AND carry no NIV/CPAP record -- the reviewer was explicit
              that a ventilator setting cannot by itself establish invasiveness.
  vasoactive  accepted infusionDrug names with rate > 0 (or a null rate, which
              cannot be ruled out), plus accepted medication orders over their
              start..stop interval.
  sedative    same rule; the reviewer already required continuous-infusion
              evidence at the drug-name level.
  crrt        continuous modalities only.
  All four    gaps of <= 1 window between two marked windows are bridged.
              A stay with no interface at all is FLAGGED, never coerced to 0.

Outputs (windows/ and review_returned/).
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
R = HERE / "review_returned"
OUT = HERE / "windows"
WINDOW = 360

coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv",
                  usecols=["patientunitstayid", "hospitalid", "max_valid_window", "n_windows"])
STAYS = set(coh.patientunitstayid)
MAXW = coh.set_index("patientunitstayid").max_valid_window
TOTALW = int(coh.n_windows.sum())
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---- frozen dictionaries from the adjudicated lists ------------------------
dr = pd.read_csv(R / "termlist_1_drugs_REVIEWED.csv", encoding="utf-8-sig")
vent = pd.read_csv(R / "termlist_3_ventilation_crrt_REVIEWED.csv", encoding="utf-8-sig")
D = {c: set(dr.loc[dr.REVIEWER_DECISION == c, "drugname"]) for c in ["vasoactive", "sedative"]}
V_INV_TREAT = set(vent.loc[(vent.REVIEWER_DECISION == "INVASIVE vent") &
                           (vent.source == "treatment.treatmentstring"), "treatmentstring"])
V_INV_AIRWAY = set(vent.loc[(vent.REVIEWER_DECISION == "INVASIVE vent") &
                            (vent.source == "respiratoryCare.airwaytype"), "treatmentstring"])
V_NIV_TREAT = set(vent.loc[(vent.REVIEWER_DECISION == "NON-invasive (exclude)") &
                           (vent.source == "treatment.treatmentstring"), "treatmentstring"])
V_NIV_CHART = set(vent.loc[(vent.REVIEWER_DECISION == "NON-invasive (exclude)") &
                           (vent.source == "respiratoryCharting label"), "treatmentstring"])
V_TIMING = set(vent.loc[vent.REVIEWER_DECISION == "vent-setting evidence", "treatmentstring"])
V_CRRT = set(vent.loc[vent.REVIEWER_DECISION == "CRRT", "treatmentstring"])
log(f"dictionaries: vasoactive {len(D['vasoactive'])}, sedative {len(D['sedative'])} drug names; "
    f"invasive {len(V_INV_TREAT)}+{len(V_INV_AIRWAY)}, NIV {len(V_NIV_TREAT)}+{len(V_NIV_CHART)}, "
    f"timing {len(V_TIMING)}, CRRT {len(V_CRRT)}")

marks, unrec = {}, {}
def mark(var, sid, w0, w1=None):
    w1 = w0 if w1 is None else w1
    mx = MAXW.get(sid)
    if mx is None: return
    for w in range(max(0, int(w0)), min(int(mx), int(w1)) + 1):
        marks.setdefault(var, set()).add((sid, w))
def note_unrec(src, values):
    unrec.setdefault(src, set()).update(values)

# ---- treatment: invasive vent, NIV, CRRT -----------------------------------
t = pd.read_csv(f"{B}/treatment.csv.gz", usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"])
t = t[t.patientunitstayid.isin(STAYS)]
seen = set(vent.treatmentstring)
pool = t[t.treatmentstring.str.contains("ventilation and oxygenation|renal\\|dialysis", regex=True, na=False)]
note_unrec("treatment.treatmentstring", set(pool.treatmentstring) - seen)
inv_stays, niv_stays = set(), set(t.loc[t.treatmentstring.isin(V_NIV_TREAT), "patientunitstayid"])
for r in t[t.treatmentstring.isin(V_INV_TREAT)].itertuples():
    inv_stays.add(r.patientunitstayid); mark("mech_vent", r.patientunitstayid, r.treatmentoffset // WINDOW)
for r in t[t.treatmentstring.isin(V_CRRT)].itertuples():
    mark("crrt", r.patientunitstayid, r.treatmentoffset // WINDOW)
log(f"treatment: {len(inv_stays):,} stays with definite-invasive evidence, {len(niv_stays):,} with NIV")

# ---- respiratoryCare: airway type and vent interval ------------------------
rc = pd.read_csv(f"{B}/respiratoryCare.csv.gz",
                 usecols=["patientunitstayid", "respcarestatusoffset", "airwaytype",
                          "ventstartoffset", "ventendoffset"], low_memory=False)
rc = rc[rc.patientunitstayid.isin(STAYS)]
for r in rc[rc.airwaytype.isin(V_INV_AIRWAY)].itertuples():
    inv_stays.add(r.patientunitstayid)
    if pd.notna(r.ventstartoffset) and r.ventstartoffset > 0 and pd.notna(r.ventendoffset) and r.ventendoffset > r.ventstartoffset:
        mark("mech_vent", r.patientunitstayid, r.ventstartoffset // WINDOW, r.ventendoffset // WINDOW)
    else:
        mark("mech_vent", r.patientunitstayid, r.respcarestatusoffset // WINDOW)
log(f"after respiratoryCare: {len(inv_stays):,} stays with definite-invasive evidence")

# ---- respiratoryCharting: timing only, and NIV exclusion -------------------
tim, chart_seen = [], set(vent.treatmentstring)
for ch in pd.read_csv(f"{B}/respiratoryCharting.csv.gz",
                      usecols=["patientunitstayid", "respchartoffset", "respcharttypecat",
                               "respchartvaluelabel"], chunksize=2_000_000, low_memory=False):
    ch = ch[ch.patientunitstayid.isin(STAYS)]
    if not len(ch): continue
    ch["key"] = ch.respcharttypecat.fillna("") + " | " + ch.respchartvaluelabel.fillna("")
    note_unrec("respiratoryCharting label", set(ch.key) - chart_seen)
    niv_stays |= set(ch.loc[ch.key.isin(V_NIV_CHART), "patientunitstayid"])
    tim.append(ch.loc[ch.key.isin(V_TIMING), ["patientunitstayid", "respchartoffset"]])
tim = pd.concat(tim, ignore_index=True) if tim else pd.DataFrame(columns=["patientunitstayid", "respchartoffset"])
use = inv_stays - niv_stays
n_before = len(tim)
tim = tim[tim.patientunitstayid.isin(use)]
for r in tim.itertuples():
    mark("mech_vent", r.patientunitstayid, r.respchartoffset // WINDOW)
log(f"respiratoryCharting timing rows: {n_before:,} total, {len(tim):,} used "
    f"({len(use):,} stays are invasive-established and NIV-free)")

# ---- drugs -----------------------------------------------------------------
inf = pd.read_csv(f"{B}/infusionDrug.csv.gz",
                  usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"], low_memory=False)
inf = inf[inf.patientunitstayid.isin(STAYS)]
note_unrec("infusionDrug.drugname", set(inf.drugname.dropna()) - set(dr.drugname))
rate = pd.to_numeric(inf.drugrate, errors="coerce")
for var, key in [("vasopressor", "vasoactive"), ("sedative", "sedative")]:
    sub = inf[inf.drugname.isin(D[key]) & (rate.isna() | (rate > 0))]
    for r in sub.itertuples():
        mark(var, r.patientunitstayid, r.infusionoffset // WINDOW)
med = pd.read_csv(f"{B}/medication.csv.gz",
                  usecols=["patientunitstayid", "drugname", "routeadmin", "drugstartoffset",
                           "drugstopoffset"], low_memory=False)
med = med[med.patientunitstayid.isin(STAYS) &
          med.routeadmin.fillna("").str.contains("iv|intraven", case=False, regex=True)]
note_unrec("medication.drugname (IV)", set(med.drugname.dropna()) - set(dr.drugname))
for var, key in [("vasopressor", "vasoactive"), ("sedative", "sedative")]:
    sub = med[med.drugname.isin(D[key])]
    for r in sub.itertuples():
        s = r.drugstartoffset if pd.notna(r.drugstartoffset) else 0
        e = r.drugstopoffset if pd.notna(r.drugstopoffset) and r.drugstopoffset > s else s
        mark(var, r.patientunitstayid, s // WINDOW, e // WINDOW)
log("drugs done")

# ---- bridge <=1-window gaps, assemble --------------------------------------
rows = [(s, w) for s in sorted(STAYS) for w in range(int(MAXW[s]) + 1)]
wide = pd.DataFrame(rows, columns=["patientunitstayid", "window_idx"])
for var in ["mech_vent", "crrt", "vasopressor", "sedative"]:
    got = marks.get(var, set())
    bridged = set(got)
    by_stay = {}
    for s, w in got: by_stay.setdefault(s, []).append(w)
    for s, ws in by_stay.items():
        ws = sorted(ws)
        for a, b in zip(ws, ws[1:]):
            if b - a == 2: bridged.add((s, a + 1))
    wide[var] = [1 if k in bridged else 0 for k in zip(wide.patientunitstayid, wide.window_idx)]
wide["vent_interface_absent"] = (~wide.patientunitstayid.isin(inv_stays | niv_stays)).astype(int)
wide.to_csv(OUT / "timewindow_ctier_eicu.csv", index=False)

# ---- report -----------------------------------------------------------------
MIMIC = {"mech_vent": 24.8, "crrt": 0.2, "vasopressor": 9.3, "sedative": 19.7}
PREV_HEURISTIC = {"mech_vent": None, "crrt": None, "vasopressor": 5.4, "sedative": 8.2}
w = wide.merge(coh[["patientunitstayid", "hospitalid"]], on="patientunitstayid")
rep = []
for var in ["mech_vent", "crrt", "vasopressor", "sedative"]:
    h = w.groupby("hospitalid")[var].mean()
    rep.append({"variable": var,
                "eicu_pct_windows": round(w[var].mean() * 100, 1),
                "mimic_pct_windows": MIMIC[var],
                "heuristic_pct_windows_before_review": PREV_HEURISTIC[var],
                "eicu_pct_stays": round(w.groupby("patientunitstayid")[var].max().mean() * 100, 1),
                "hospitals_with_any": int((h > 0).sum()),
                "hospitals_total": int(h.size)})
rep = pd.DataFrame(rep)
rep.to_csv(R / "ctier_capture_check.csv", index=False)
print("\n=== capture prevalence, reviewed dictionaries ===")
print(rep.to_string(index=False))
print(f"\nstays with no ventilation interface at all (flagged, not coerced to 0): "
      f"{wide.groupby('patientunitstayid').vent_interface_absent.max().sum():,} of {len(STAYS):,}")

ur = pd.DataFrame([{"source": k, "n_unrecognised_terms": len(v),
                    "examples": "; ".join(sorted(v)[:5])} for k, v in unrec.items()])
ur.to_csv(R / "unrecognised_terms.csv", index=False)
print("\n=== unrecognised terms (present in eICU, below the frequency cut when the lists were built) ===")
print(ur.to_string(index=False))
log("done")
