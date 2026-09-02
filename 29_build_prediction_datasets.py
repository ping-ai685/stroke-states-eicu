"""
Paper 2, Aim 4 (added 2026-08-30, after the Aim 1-3 transport results were
known -- see protocol v1.3): build the one-step-ahead prediction datasets.

The task is: given everything observable through 6-hour window t, which state
does the patient occupy in window t+1?

Two target definitions are built and both are carried forward, because they
answer different questions and the null models differ:

  4-class   among window pairs where a successor window exists. "Which state
            next, given the patient is still in the ICU."
  5-class   adds `_end` for stays that actually terminated. Administrative
            censoring at the 72 h boundary is NOT an event and those windows
            are dropped from the 5-class target rather than labelled `_end`.

Nothing is fitted here. This script only assembles aligned matrices and reports
the persistence rate that any model has to beat.

Output: prediction/{mimic,eicu}_pairs.csv plus prediction/dataset_qa.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
T1 = ROOT / "04_outputs/tables"
OUT = HERE / "prediction"; OUT.mkdir(exist_ok=True)

FEATS = pd.read_csv(HERE / "frozen_params/feature_order.csv").feature.tolist()

# Two different orderings live in this project and confusing them silently
# scrambles the MIMIC labels: the fitted model's own state index is `raw_state`
# (that is what step 19 maps eICU decodings through), but the stored MIMIC
# assignments in hmm_state_assignments_all.csv are written in `canonical_state`
# order. The prevalence assertion below exists because that mistake is otherwise
# invisible -- every state still gets a valid name, just the wrong one.
_LM = pd.read_csv(T1 / "state_label_mapping.csv")
CANON = _LM.set_index("canonical_state")["name"].to_dict()
PUBLISHED_PCT = {"neurologically preserved-low support": 63.3,
                 "neurological impairment-respiratory support": 17.1,
                 "neurological impairment-renal dysfunction": 11.8,
                 "neurological impairment-low support": 7.8}


def make_pairs(w, pid, state_col, horizon_last):
    """Attach the next window's state. Flags distinguish a real stay end from
    administrative censoring at the end of the 72 h observation period."""
    w = w.sort_values([pid, "window_idx"]).reset_index(drop=True)
    w["next_state"] = w.groupby(pid)[state_col].shift(-1)
    w["next_window_idx"] = w.groupby(pid)["window_idx"].shift(-1)
    last = w.groupby(pid)["window_idx"].transform("max")
    w["is_last_window"] = w.window_idx == last
    # contiguity: a successor must be the immediately following window
    w["successor_contiguous"] = w.next_window_idx == w.window_idx + 1
    # a stay that stops before the observation horizon truly ended
    w["stay_ended"] = w.is_last_window & (last < horizon_last)
    w["censored_at_horizon"] = w.is_last_window & (last >= horizon_last)
    return w


rows = []

# ---- MIMIC-IV (training and internal test) -----------------------------------
mm = pd.read_csv(T1 / "timewindow_level_modeling.csv")
st = pd.read_csv(T1 / "hmm_state_assignments_all.csv")
mm = mm.merge(st, on=["stay_id", "window_idx"], how="inner", suffixes=("", "_st"))
missing = [f for f in FEATS if f not in mm.columns]
assert not missing, f"MIMIC missing features: {missing}"
mm["state"] = mm.state_k4.map(CANON)
assert mm.state.notna().all()
got = mm.state.value_counts(normalize=True) * 100
for name, want in PUBLISHED_PCT.items():
    assert abs(got[name] - want) < 0.15, (
        f"MIMIC state prevalence for {name!r} is {got[name]:.1f}%, published {want}% "
        "-- the state label mapping is wrong")
print("MIMIC state prevalences match the published values; label mapping verified")
H = int(mm.window_idx.max())
mm = make_pairs(mm, "stay_id", "state", H)
mm["cohort"] = "mimic"
mm = mm.rename(columns={"stay_id": "uid"})

# ---- eICU (external test) ----------------------------------------------------
cont = pd.read_csv(HERE / "windows/timewindow_level_modeling_eicu_A1.csv")
bina = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv")
ee = cont.merge(bina, on=["patientunitstayid", "window_idx"], how="left")
ea = pd.read_csv(HERE / "results_full_model/eicu_state_assignments_full_A1.csv")
ee = ee.merge(ea[["patientunitstayid", "window_idx", "state", "hospitalid",
                  "vent_ascertainable"]], on=["patientunitstayid", "window_idx"], how="inner")
missing = [f for f in FEATS if f not in ee.columns]
assert not missing, f"eICU missing features: {missing}"
assert int(ee.window_idx.max()) == H, "observation horizon differs between databases"
ee = make_pairs(ee, "patientunitstayid", "state", H)
ee["vent_ascertainable"] = ee.vent_ascertainable.astype(bool)   # 0/1 would index columns
elig = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
ee["gcs_eligible"] = ee.patientunitstayid.isin(elig)
ee["cohort"] = "eicu"
ee = ee.rename(columns={"patientunitstayid": "uid"})

KEEP_M = ["uid", "window_idx", "cohort", "split", "state", "next_state",
          "successor_contiguous", "stay_ended", "censored_at_horizon"] + FEATS
KEEP_E = ["uid", "window_idx", "cohort", "state", "next_state", "hospitalid",
          "vent_ascertainable", "gcs_eligible",
          "successor_contiguous", "stay_ended", "censored_at_horizon"] + FEATS
mm[KEEP_M].to_csv(OUT / "mimic_pairs.csv", index=False)
ee[KEEP_E].to_csv(OUT / "eicu_pairs.csv", index=False)

# ---- QA: the persistence rate every model must beat --------------------------
def qa(df, label, extra=None):
    sel = df[df.successor_contiguous]
    same = (sel.state == sel.next_state)
    n5 = df[df.successor_contiguous | df.stay_ended]
    rows.append({"scope": label, "patients": df.uid.nunique(),
                 "windows": len(df), "pairs_4class": len(sel),
                 "persistence_pct": round(same.mean() * 100, 2),
                 "changed_pairs": int((~same).sum()),
                 "changed_pct": round((~same).mean() * 100, 2),
                 "pairs_5class": len(n5),
                 "end_events": int(df.stay_ended.sum()),
                 "censored_at_horizon": int(df.censored_at_horizon.sum())})

qa(mm[mm.split == "train"], "mimic-train")
qa(mm[mm.split == "test"], "mimic-test")
qa(ee, "eicu-full")
qa(ee[ee.gcs_eligible], "eicu-gcs-eligible")
qa(ee[ee.vent_ascertainable], "eicu-vent-ascertainable")
qa(ee[ee.gcs_eligible & ee.vent_ascertainable], "eicu-both")
q = pd.DataFrame(rows)
q.to_csv(OUT / "dataset_qa.csv", index=False)
print(q.to_string(index=False))

print("\nper-state persistence, eICU full cohort:")
sel = ee[ee.successor_contiguous]
for s, g in sel.groupby("state"):
    print(f"  {s:<48} {(g.state == g.next_state).mean()*100:5.1f}%   n={len(g):,}")
print(f"\nwrote {OUT/'mimic_pairs.csv'} and {OUT/'eicu_pairs.csv'}")
