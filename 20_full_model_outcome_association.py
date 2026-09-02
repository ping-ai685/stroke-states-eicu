"""
Paper 2, Stage C for the full model: outcome and association analyses.

Mirrors 03_code/10_outcome_association.py exactly in construction -- same three
window-level outcomes, same GEE with an exchangeable working correlation
clustered by patient, same restriction of each onset outcome to windows not
already receiving that support, same 10-event suppression rule.

Two protocol-sanctioned deviations, both stated rather than worked around:

  Charlson  eICU ICD coding is shallower than MIMIC's, so a recomputed Charlson
            would not be comparable. Protocol section 12.2 provides for exactly
            this: the primary model uses the reduced common covariate set (age,
            sex, stroke subtype) and APACHE IV enters only a clearly labelled
            sensitivity model, never the primary.
  Outcomes  No post-discharge survival exists in eICU. ICU and hospital
            mortality only.

Reminder: the column is `vasopressor`; the variable is VASOACTIVE SUPPORT.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results_full_model"
MIN_EVENTS = 10
REF = "neurologically preserved-low support"

a = pd.read_csv(OUT / "eicu_state_assignments_full_A1.csv")
b = pd.read_csv(HERE / "windows/timewindow_ctier_eicu.csv")
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
elig = set(pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv").patientunitstayid)

df = a.merge(b, on=["patientunitstayid", "window_idx"]).merge(
    coh[["patientunitstayid", "age_n", "gender", "stroke_subtype", "icu_mortality",
         "hospital_mortality", "unitdischargeoffset"]],
    on="patientunitstayid").sort_values(["patientunitstayid", "window_idx"]).reset_index(drop=True)
df["left_icu_within_72h"] = df.unitdischargeoffset < 4320
df["state"] = pd.Categorical(df.state)

mv, vp, dh = [], [], []
for sid, g in df.groupby("patientunitstayid", sort=False):
    g = g.sort_values("window_idx")
    m1, m2 = g.mech_vent.shift(-1), g.mech_vent.shift(-2)
    v1, v2 = g.vasopressor.shift(-1), g.vasopressor.shift(-2)
    mv.append(((m1 == 1) | (m2 == 1)).astype(float).where(~(m1.isna() & m2.isna())))
    vp.append(((v1 == 1) | (v2 == 1)).astype(float).where(~(v1.isna() & v2.isna())))
    last = g.window_idx == g.window_idx.max()
    dh.append((last & (g.icu_mortality == 1) & g.left_icu_within_72h).astype(int))
df["mv_onset_12h"] = pd.concat(mv).values
df["vp_onset_12h"] = pd.concat(vp).values
df["icu_death_hazard"] = pd.concat(dh).values

FORM = "{o} ~ C(state, Treatment(reference='%s')) + age_n + C(gender) + C(stroke_subtype, Treatment(reference='AIS'))" % REF
FORM_CRUDE = "{o} ~ C(state, Treatment(reference='%s'))" % REF

def run(data, outcome, scope, label):
    if data[outcome].sum() < MIN_EVENTS:
        return []
    out = []
    for tag, f in [("adjusted", FORM), ("crude", FORM_CRUDE)]:
        try:
            r = smf.gee(f.format(o=outcome), groups="patientunitstayid", data=data,
                        family=sm.families.Binomial()).fit()
        except Exception as e:
            print(f"  {label}/{tag}: model failed ({type(e).__name__})"); continue
        for term in [t for t in r.params.index if t.startswith("C(state")]:
            st = term.split("T.")[-1].rstrip("]")
            n_ev = int(data.loc[data.state == st, outcome].sum())
            out.append({"scope": scope, "outcome": label, "model": tag, "state": st,
                        "n_events_in_state": n_ev,
                        "OR": round(float(np.exp(r.params[term])), 2),
                        "CI_low": round(float(np.exp(r.conf_int().loc[term, 0])), 2),
                        "CI_high": round(float(np.exp(r.conf_int().loc[term, 1])), 2),
                        "suppressed_lt10_events": n_ev < MIN_EVENTS})
    return out

rows = []
for scope, sub in [("full", df), ("both-eligible", df[df.patientunitstayid.isin(elig) & (df.vent_ascertainable == 1)])]:
    sets = [("new invasive ventilation within 12 h", sub[(sub.mech_vent == 0) & sub.mv_onset_12h.notna()], "mv_onset_12h"),
            ("new vasoactive support within 12 h", sub[(sub.vasopressor == 0) & sub.vp_onset_12h.notna()], "vp_onset_12h"),
            ("ICU death (discrete-time hazard)", sub, "icu_death_hazard")]
    print(f"\n=== {scope} ===")
    for label, data, oc in sets:
        print(f"{label}: {len(data):,} windows, {data.patientunitstayid.nunique():,} patients, "
              f"event rate {data[oc].mean()*100:.2f}%")
        rows += run(data, oc, scope, label)
res = pd.DataFrame(rows)
res.to_csv(OUT / "outcome_association_full.csv", index=False)

print("\n=== adjusted odds ratios, reference = neurologically preserved-low support ===")
for scope in res.scope.unique():
    print(f"\n-- {scope} --")
    t = res[(res.scope == scope) & (res.model == "adjusted")]
    for oc in t.outcome.unique():
        print(f"  {oc}")
        for _, r in t[t.outcome == oc].iterrows():
            flag = "  [SUPPRESSED, <10 events]" if r.suppressed_lt10_events else ""
            print(f"    {r.state:<44} OR {r.OR:>6.2f}  ({r.CI_low}-{r.CI_high})  n_ev={r.n_events_in_state}{flag}")

# mortality gradient by last state, matched construction
last = df.sort_values(["patientunitstayid", "window_idx"]).groupby("patientunitstayid").tail(1)
g = last.groupby("state").agg(n=("patientunitstayid", "size"),
                              icu_death_pct=("icu_mortality", lambda s: round(s.mean()*100, 1)),
                              hosp_death_pct=("hospital_mortality", lambda s: round(s.mean()*100, 1)))
MIMIC = {"neurologically preserved-low support": 2.9, "neurological impairment-low support": 3.8,
         "neurological impairment-renal dysfunction": 28.0, "neurological impairment-respiratory support": 43.9}
g["mimic_icu_death_pct"] = [MIMIC[i] for i in g.index]
g.to_csv(OUT / "outcome_gradient_full.csv")
print("\n=== mortality by last observed state (full cohort, 21-variable model) ===")
print(g.to_string())
rho = g.icu_death_pct.corr(g.mimic_icu_death_pct, method="spearman")
print(f"rank order reproduced: {'YES' if (g.icu_death_pct.rank()==g.mimic_icu_death_pct.rank()).all() else 'NO'}"
      f"   Spearman rho = {rho:.3f}   max abs diff {abs(g.icu_death_pct-g.mimic_icu_death_pct).max():.1f} pp")
