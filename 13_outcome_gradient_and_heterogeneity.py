"""
Paper 2, step 10 (partial): construct validity and hospital heterogeneity for the
treatment-free control.

Protocol v1.1 section 10.4 lists "construct / prognostic validity" as a primary
validation domain, and section 14 covers hospital heterogeneity. Both are Stage C
and both are runnable now; neither needs the four organ-support variables.

Outcome-blind harmonization (section 19) is complete and the state assignments
are archived, so outcome fields may be read from here.

SCOPE LIMIT, stated rather than worked around: section 12.1's support-escalation
outcomes -- new invasive ventilation and new vasopressor within 12 h -- require
the C-tier variables and are deferred with the full model. Only ICU and hospital
mortality are available now, and eICU has no post-discharge survival at all.

Also runs the amendment-I confound test that the PI approved for v1.2: does
hospital-level preserved-state prevalence track hospital-level GCS coverage?

Outputs (results_treatment_free/).
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / "results_treatment_free"

a = pd.read_csv(OUT / "eicu_state_assignments_A1.csv")
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
elig = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
SHORT = {s: s.replace(" analogue", "") for s in a.state.unique()}

# ---- last observed state per patient, the discovery study's construction -----
last = a.sort_values(["patientunitstayid", "window_idx"]).groupby("patientunitstayid").tail(1)
d = last[["patientunitstayid", "state"]].merge(
    coh[["patientunitstayid", "hospitalid", "icu_mortality", "hospital_mortality", "n_windows"]],
    on="patientunitstayid")
d["state"] = d.state.map(SHORT)

MIMIC_LAST = {"neurologically preserved-low support": 2.9,
              "neurological impairment-low support": 3.8,
              "neurological impairment-renal dysfunction": 28.0,
              "neurological impairment-respiratory support": 43.9}
rows = []
for scope, sub in [("full", d), ("eligible", d[d.patientunitstayid.isin(elig)])]:
    g = sub.groupby("state").agg(n=("patientunitstayid", "size"),
                                 icu_death_pct=("icu_mortality", lambda s: round(s.mean()*100, 1)),
                                 hosp_death_pct=("hospital_mortality", lambda s: round(s.mean()*100, 1)))
    g["scope"] = scope
    g["mimic_icu_death_pct_primary_model"] = [MIMIC_LAST[i] for i in g.index]
    rows.append(g.reset_index())
grad = pd.concat(rows)
grad.to_csv(OUT / "outcome_gradient.csv", index=False)
print("=== mortality by LAST observed state (eICU, treatment-free A1) ===")
print("MIMIC column is the PRIMARY model's published gradient -- indicative, not a matched target,")
print("because the treatment-free states are analogues rather than the same objects.\n")
for scope in ["full", "eligible"]:
    g = grad[grad.scope == scope].set_index("state")
    print(f"--- {scope} ---")
    print(g[["n", "icu_death_pct", "hosp_death_pct", "mimic_icu_death_pct_primary_model"]].to_string())
    order_e = g.icu_death_pct.rank(ascending=False).astype(int)
    order_m = g.mimic_icu_death_pct_primary_model.rank(ascending=False).astype(int)
    rho = pd.Series(g.icu_death_pct).corr(pd.Series(g.mimic_icu_death_pct_primary_model), method="spearman")
    print(f"    rank order reproduced: {'YES' if (order_e == order_m).all() else 'NO'}   "
          f"Spearman rho = {rho:.3f}\n")

# ---- amendment I confound test ---------------------------------------------
w = pd.read_csv(HERE / "windows/timewindow_level_modeling_eicu_A1.csv",
                usecols=["patientunitstayid", "window_idx", "gcs_motor_missing_raw"])
w = w.merge(a[["patientunitstayid", "window_idx", "state"]], on=["patientunitstayid", "window_idx"])
w = w.merge(coh[["patientunitstayid", "hospitalid"]], on="patientunitstayid")
w["preserved"] = (w.state == "neurologically preserved-low support analogue").astype(int)
res = []
for scope, sub in [("full", w), ("eligible", w[w.patientunitstayid.isin(elig)])]:
    h = sub.groupby("hospitalid").agg(n_windows=("preserved", "size"),
                                      gcs_coverage=("gcs_motor_missing_raw", lambda s: 1 - s.mean()),
                                      preserved_prev=("preserved", "mean"))
    h = h[h.n_windows >= 100]
    r = h["gcs_coverage"].corr(h["preserved_prev"])
    # slope in percentage points of preserved prevalence per 10pp of GCS coverage
    b = np.polyfit(h.gcs_coverage, h.preserved_prev, 1)[0]
    res.append({"scope": scope, "hospitals": len(h), "corr_gcs_coverage_vs_preserved_prev": round(r, 3),
                "pp_change_in_preserved_per_10pp_gcs": round(b * 10, 1),
                "preserved_prev_range": f"{h.preserved_prev.min()*100:.0f}-{h.preserved_prev.max()*100:.0f}%"})
    h.to_csv(OUT / f"hospital_preserved_vs_gcs_{scope}.csv")
ct = pd.DataFrame(res); ct.to_csv(OUT / "amendment_I_confound_test.csv", index=False)
print("=== amendment I confound test: hospital GCS coverage vs preserved-state prevalence ===")
print(ct.to_string(index=False))
print("\nA negative correlation means hospitals that chart less GCS show more preserved-state")
print("windows -- i.e. part of the Aim 3 between-hospital signal is measurement, not care.")
print(f"\nWrote {OUT}/")
