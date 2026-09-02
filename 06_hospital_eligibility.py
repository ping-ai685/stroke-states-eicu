"""
Paper 2, step 6: freeze the hospital eligibility list.

Rule approved by the PI on 23 August 2026, from the tier table in section H2 of
PROTOCOL_REVIEW_v1.0.md: a hospital enters the hospital-level analyses if it
contributes >=20 eligible stroke patients AND charts GCS in >=50% of their
6-hour windows.

GCS drives the rule rather than patient count because GCS is the binding
constraint: 53.5% of windows against MIMIC's 94.9%, and median-imputing a
missing GCS assigns eye=4/motor=6 -- "normal" -- which pushes unmeasured windows
toward the neurologically-preserved state. A hospital that does not chart GCS
cannot contribute an interpretable state assignment, however many patients it has.

The patient-level external validation is still reported on the FULL cohort; the
eligible subset is the pre-specified primary sensitivity analysis and the
denominator for every hospital-level (Aim 3) result.

Outputs (cohort/):
  hospital_eligibility_frozen.csv   one row per hospital, with the decision
  patient_level_cohort_eligible.csv the eligible subset
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
MIN_PATIENTS, MIN_GCS_FRAC = 20, 0.50
FREEZE_DATE = "2026-08-23"

coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
gcs = pd.read_csv(HERE / "audit/gcs_coverage_by_stay.csv",
                  usecols=["patientunitstayid", "gcs_windows", "gcs_frac"])
coh = coh.merge(gcs, on="patientunitstayid", how="left")

h = coh.groupby("hospitalid").agg(
    n_patients=("patientunitstayid", "size"),
    n_windows=("n_windows", "sum"),
    gcs_window_coverage=("gcs_frac", "mean"),
    pct_no_gcs_at_all=("gcs_frac", lambda s: (s == 0).mean()),
    icu_mortality=("icu_mortality", "mean"),
).reset_index()
for c in ["numbedscategory", "teachingstatus", "region"]:
    h = h.merge(coh.groupby("hospitalid")[c].first().reset_index(), on="hospitalid", how="left")

h["meets_n"] = (h.n_patients >= MIN_PATIENTS).astype(int)
h["meets_gcs"] = (h.gcs_window_coverage >= MIN_GCS_FRAC).astype(int)
h["eligible"] = (h.meets_n & h.meets_gcs).astype(int)
h["exclusion_reason"] = [
    "" if e else ("patients<20 and GCS<50%" if not n and not g else
                  ("patients<20" if not n else "GCS coverage<50%"))
    for e, n, g in zip(h.eligible, h.meets_n, h.meets_gcs)]
h["freeze_date"] = FREEZE_DATE
h.sort_values(["eligible", "n_patients"], ascending=[False, False]) \
 .to_csv(HERE / "cohort/hospital_eligibility_frozen.csv", index=False)

elig = coh[coh.hospitalid.isin(h.loc[h.eligible == 1, "hospitalid"])]
elig.to_csv(HERE / "cohort/patient_level_cohort_eligible.csv", index=False)

print(f"Rule: n>={MIN_PATIENTS} patients AND GCS in >={MIN_GCS_FRAC:.0%} of windows  (frozen {FREEZE_DATE})\n")
print(f"{'':<26}{'hospitals':>10}{'patients':>10}{'windows':>10}")
print(f"{'full cohort':<26}{len(h):>10}{len(coh):>10,}{coh.n_windows.sum():>10,}")
print(f"{'eligible':<26}{h.eligible.sum():>10}{len(elig):>10,}{elig.n_windows.sum():>10,}")
print(f"{'':<26}{'':>10}{len(elig)/len(coh)*100:>9.0f}%{elig.n_windows.sum()/coh.n_windows.sum()*100:>9.0f}%")
print("\nExclusions:")
print(h[h.eligible == 0].exclusion_reason.value_counts().to_string())

print(f"\nEligible cohort vs full cohort:")
for lbl, d in [("full", coh), ("eligible", elig)]:
    print(f"  {lbl:<10} n={len(d):>5,}  age {d.age_n.median():.0f}  "
          f"ICU mort {d.icu_mortality.mean()*100:4.1f}%  "
          f"GCS windows {d.gcs_frac.mean()*100:4.1f}%  "
          f"AIS {(d.stroke_subtype=='AIS').mean()*100:4.1f}%  "
          f"NeuroICU {(d.unittype=='Neuro ICU').mean()*100:4.1f}%")
print("\nEligible hospitals by characteristics:")
he = h[h.eligible == 1]
for c in ["teachingstatus", "numbedscategory", "region"]:
    print(f"  {c}: " + ", ".join(f"{k}={v}" for k, v in he[c].value_counts(dropna=False).items()))
