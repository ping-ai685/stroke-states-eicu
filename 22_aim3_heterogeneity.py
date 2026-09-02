"""
Paper 2, Aim 3: between-hospital heterogeneity of the frozen-model states.

Protocol section 14, restricted to the hospitals that pass both eligibility rules.
Uses the frozen A1 assignment, so nothing here depends on the de novo refit.

Amendment I requires GCS window coverage to enter the hospital-level models as a
covariate rather than being assumed away: in the full cohort the correlation
between a hospital's GCS coverage and its preserved-state prevalence is -0.699,
which the eligibility rule reduces to -0.199. Small, not zero.

Between-hospital variance is estimated with a random-intercept linear probability
model. That is a deliberate choice over a random-intercept logistic: with 39
clusters the logistic variance component is poorly identified, while the LPM
variance is on the probability scale and reads directly as "how many percentage
points do hospitals differ by". The ICC is reported on the same scale.

Outputs (results_aim3/).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results_aim3"; OUT.mkdir(exist_ok=True)

froz = pd.read_csv(HERE / "results_full_model/eicu_state_assignments_full_A1.csv")
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
elig = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)
gcs = pd.read_csv(HERE / "audit/gcs_coverage_by_stay.csv", usecols=["patientunitstayid", "gcs_frac"])
keep = elig & set(coh.loc[coh.vent_ascertainable == 1, "patientunitstayid"])
d = froz[froz.patientunitstayid.isin(keep)].merge(gcs, on="patientunitstayid").merge(
    coh[["patientunitstayid", "numbedscategory", "teachingstatus", "region", "unittype"]],
    on="patientunitstayid")
STATES = sorted(d.state.unique())
print(f"Aim 3 analysis set: {d.hospitalid.nunique()} hospitals, "
      f"{d.patientunitstayid.nunique():,} patients, {len(d):,} windows")

# ---- hospital-level prevalence and between-hospital spread ------------------
h = d.groupby("hospitalid").agg(n_windows=("state", "size"),
                                n_patients=("patientunitstayid", "nunique"),
                                gcs_coverage=("gcs_frac", "mean")).reset_index()
for s in STATES:
    h[s] = d.assign(x=(d.state == s).astype(int)).groupby("hospitalid").x.mean().values
for c in ["numbedscategory", "teachingstatus", "region", "unittype"]:
    h[c] = d.groupby("hospitalid")[c].agg(lambda x: x.mode().iat[0] if len(x.mode()) else np.nan).values
h.to_csv(OUT / "hospital_state_prevalence.csv", index=False)

rows = []
for s in STATES:
    d["y"] = (d.state == s).astype(int)
    try:
        r = smf.mixedlm("y ~ 1", d, groups=d.hospitalid).fit(reml=True)
        vu = float(r.cov_re.iloc[0, 0]); ve = float(r.scale)
        icc = vu / (vu + ve)
        sd_between = np.sqrt(vu)
    except Exception as e:
        icc, sd_between = np.nan, np.nan
        print(f"  mixedlm failed for {s}: {type(e).__name__}")
    rows.append({"state": s,
                 "pooled_pct": round(d.y.mean() * 100, 1),
                 "hospital_min_pct": round(h[s].min() * 100, 1),
                 "hospital_median_pct": round(h[s].median() * 100, 1),
                 "hospital_max_pct": round(h[s].max() * 100, 1),
                 "between_hospital_SD_pp": round(sd_between * 100, 1),
                 "ICC": round(icc, 3)})
het = pd.DataFrame(rows); het.to_csv(OUT / "between_hospital_variance.csv", index=False)
print("\n=== between-hospital heterogeneity of state prevalence ===")
print(het.to_string(index=False))

# ---- hospital-level regression, with GCS coverage as a covariate ------------
reg_rows = []
for s in STATES:
    hh = h.dropna(subset=["teachingstatus", "numbedscategory", "region"]).copy()
    hh["y"] = hh[s] * 100
    f = ("y ~ gcs_coverage + C(teachingstatus) + C(numbedscategory) + C(region)")
    try:
        r = smf.ols(f, data=hh).fit()
        for term in r.params.index:
            if term == "Intercept":
                continue
            reg_rows.append({"state": s, "term": term,
                             "coef_pp": round(float(r.params[term]), 2),
                             "CI_low": round(float(r.conf_int().loc[term, 0]), 2),
                             "CI_high": round(float(r.conf_int().loc[term, 1]), 2),
                             "p": round(float(r.pvalues[term]), 3)})
        if s == "neurologically preserved-low support":
            print(f"\npreserved-state model: n={int(r.nobs)} hospitals, adj R2={r.rsquared_adj:.3f}")
    except Exception as e:
        print(f"  ols failed for {s}: {type(e).__name__}")
reg = pd.DataFrame(reg_rows); reg.to_csv(OUT / "hospital_covariate_regression.csv", index=False)
print("\n=== hospital-level covariates, effect on state prevalence (percentage points) ===")
print("GCS coverage is expressed per 1.0 = 100 percentage points of coverage.")
print(reg[reg.term.str.contains("gcs_coverage")].to_string(index=False))
print("\nother covariates reaching p<0.05:")
sig = reg[(reg.p < 0.05) & (~reg.term.str.contains("gcs_coverage"))]
print(sig.to_string(index=False) if len(sig) else "  none")

# ---- residual confound check ------------------------------------------------
c = h["gcs_coverage"].corr(h["neurologically preserved-low support"])
print(f"\nresidual corr(GCS coverage, preserved-state prevalence) in the Aim 3 set: {c:+.3f}")
pd.DataFrame([{"corr_gcs_vs_preserved": round(c, 3), "hospitals": len(h)}]).to_csv(
    OUT / "amendment_I_residual_check.csv", index=False)
