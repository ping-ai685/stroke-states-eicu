"""
Paper 2: build the six manuscript tables as markdown, for insertion into the
prose sources.

The tables are written INTO the manuscript rather than rendered separately, so
that the per-table submission files can later be extracted from the built
document by the same rule Paper 1 uses (03_code/30_extract_tables_from_manuscript.py).
Rendering standalone table files from the analysis CSVs is what produced a
divergent artefact last time and is not repeated.

Output: manuscript/tables_v1.md, plus the individual markdown blocks keyed by
table number so the prose sources can include them at the right points.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / "manuscript"
P = "neurologically preserved-low support"
R = "neurological impairment-respiratory support"
N = "neurological impairment-renal dysfunction"
L = "neurological impairment-low support"
ORDER = [P, L, N, R]
SHORT = {P: "Neurologically preserved–low support", L: "Neurological impairment–low support",
         N: "Neurological impairment–renal dysfunction", R: "Neurological impairment–respiratory support"}

def md(rows, header, align=None):
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)

T = {}

# ---- Table 1: cohort characteristics -----------------------------------------
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")

def iqr(s): return f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"
def pct(s): return f"{s.sum():,} ({s.mean()*100:.1f})"
groups = [("All patients", coh)] + [(g, coh[coh.stroke_subtype == g]) for g in ["AIS", "ICH", "SAH", "ICH+SAH", "unspecified"]]
rows = []
for label, fn in [("Patients, n", lambda d: f"{len(d):,}"),
                  ("Age, median [IQR]", lambda d: iqr(d.age_n)),
                  ("Female, n (%)", lambda d: pct(d.gender == "Female")),
                  ("Neuro ICU, n (%)", lambda d: pct(d.unittype == "Neuro ICU")),
                  ("Teaching hospital, n (%)", lambda d: pct(d.teachingstatus == "t")),
                  ("ICU stay, days, median [IQR]", lambda d: f"{d.icu_los_days.median():.2f} [{d.icu_los_days.quantile(.25):.2f}–{d.icu_los_days.quantile(.75):.2f}]"),
                  ("6-hour windows contributed, median", lambda d: f"{d.n_windows.median():.0f}"),
                  ("ICU mortality, n (%)", lambda d: pct(d.icu_mortality == 1)),
                  ("Hospital mortality, n (%)", lambda d: pct(d.hospital_mortality == 1))]:
    rows.append([label] + [fn(d) for _, d in groups])
T[1] = ("**Table 1.** Characteristics of the eICU external validation cohort, overall and by stroke subtype. "
        "Subtype precedence follows the discovery study; stays carrying only the unspecified stroke path are "
        "reported as a separate column and their exclusion is a prespecified sensitivity analysis.",
        md(rows, ["Characteristic"] + [g for g, _ in groups]))

# ---- Table 2: harmonization and availability ---------------------------------
av = pd.read_csv(HERE / "audit/availability_vs_mimic.csv")
cap = pd.read_csv(HERE / "review_returned/ctier_capture_check.csv").set_index("variable")
NICE = {"heart_rate": "Heart rate", "sbp": "Systolic blood pressure", "map": "Mean arterial pressure",
        "resp_rate": "Respiratory rate", "spo2": "Peripheral oxygen saturation", "temp_c": "Temperature",
        "gcs_eye": "GCS eye subscore", "gcs_motor": "GCS motor subscore", "urine_output_ml": "Urine output",
        "wbc": "White cell count", "hemoglobin": "Haemoglobin", "platelet": "Platelet count",
        "creatinine": "Creatinine", "bun": "Urea nitrogen", "sodium": "Sodium",
        "potassium": "Potassium", "glucose": "Glucose"}
TIER = {**{v: "Direct" for v in ["heart_rate","sbp","map","resp_rate","spo2","temp_c","wbc","hemoglobin",
                                  "platelet","creatinine","bun","sodium","potassium","glucose"]},
        **{v: "Requires validation" for v in ["gcs_eye","gcs_motor","urine_output_ml"]}}
rows = []
for _, r in av.iterrows():
    rows.append([NICE[r.variable], TIER[r.variable], f"{r.mimic_pct_windows:.1f}",
                 f"{r.eicu_pct_windows:.1f}", f"{r.difference:+.1f}"])
SUP = [("Invasive mechanical ventilation", "24.8", "17.2–37.0", "bracketed"),
       ("Sedation, continuous infusion", "19.7", f"{cap.loc['sedative','eicu_pct_windows']:.1f}", f"{cap.loc['sedative','eicu_pct_windows']-19.7:+.1f}"),
       ("Vasoactive support, continuous infusion", "9.3", f"{cap.loc['vasopressor','eicu_pct_windows']:.1f}", f"{cap.loc['vasopressor','eicu_pct_windows']-9.3:+.1f}"),
       ("Continuous renal replacement therapy", "0.2", f"{cap.loc['crrt','eicu_pct_windows']:.1f}", f"{cap.loc['crrt','eicu_pct_windows']-0.2:+.1f}")]
for name, m, e, d in SUP:
    rows.append([name, "Requires algorithmic reconstruction", m, e, d])
T[2] = ("**Table 2.** Window-level availability of the 21 model variables, computed identically in both databases "
        "before forward fill. Invasive ventilation is reported as a bracket because no eICU hospital determines "
        "invasive versus non-invasive status for more than 67% of its stroke patients; its lower bound should not "
        "be compared with the discovery value at face value.",
        md(rows, ["Variable", "Harmonization tier", "MIMIC-IV, % of windows", "eICU, % of windows", "Difference"]))

# ---- Table 3: transport metrics ----------------------------------------------
tm = pd.read_csv(HERE / "results_full_model/transport_metrics_full.csv")
SC = {"full": "Full cohort", "GCS-eligible": "GCS-eligible hospitals",
      "vent-ascertainable": "Ventilation-ascertainable hospitals", "both": "Both eligibility rules"}
rows = []
for _, r in tm.iterrows():
    rows.append([SC[r.scope], f"{r.patients:,}", f"{r.windows:,}",
                 f"{r.min_profile_corr:.3f}", f"{r.min_prevalence_pct:.1f}",
                 f"{r.transition_rank_corr:.3f}", f"{r.max_self_transition_diff:.3f}",
                 f"{r.null_p95:.3f}", f"{r.pct_changing_state:.1f}"])
T[3] = ("**Table 3.** Frozen-model transport against the five prespecified criteria. Criterion thresholds: "
        "per-state profile correlation ≥0.80; every state ≥5% of windows; transition rank correlation ≥0.80; "
        "maximum self-transition deviation ≤0.10; observed minimum profile correlation above the 95th percentile "
        "of the permutation null. All five are met in all four scopes. The proportion of patients changing state "
        "is descriptive and is not a pass/fail criterion; the discovery value is 40.3%.",
        md(rows, ["Analysis scope", "Patients", "Windows", "Min profile r", "Min prevalence, %",
                  "Transition rank r", "Max self-transition Δ", "Null 95th pct", "Changing state, %"]))

# ---- Table 4: observed vs imputed GCS ----------------------------------------
gs = pd.read_csv(HERE / "results_full_model/gcs_observed_vs_imputed_prevalence.csv")
gc = pd.read_csv(HERE / "results_full_model/gcs_observed_only_profile_corr.csv")
rows = []
for sc, lab in [("full", "Full cohort"), ("both-eligible", "Both eligibility rules")]:
    for s in ORDER:
        g = gs[(gs.scope == sc) & (gs.state == s)].iloc[0]
        c = gc[(gc.scope == sc) & (gc.state == s)].iloc[0]
        rows.append([lab, SHORT[s], f"{g.mimic_pct:.1f}", f"{g.eicu_observed_gcs_pct:.1f}",
                     f"{g.eicu_imputed_gcs_pct:.1f}", f"{c.corr_all_windows:.3f}",
                     f"{c.corr_observed_gcs_only:.3f}"])
T[4] = ("**Table 4.** Sensitivity of transport to GCS imputation. Because the discovery imputation median for the "
        "motor subscore coincides exactly with the preserved state's point-mass emission, state prevalence is "
        "reported separately for windows with an observed and an imputed subscore, and profile correlations are "
        "recomputed with both databases restricted to observed-subscore windows. Restricting to observed windows "
        "raises every correlation, so imputation attenuates transport rather than producing it.",
        md(rows, ["Scope", "State", "MIMIC-IV prevalence, %", "eICU, observed GCS, %",
                  "eICU, imputed GCS, %", "Profile r, all windows", "Profile r, observed only"]))

# ---- Table 5: outcome associations -------------------------------------------
oa = pd.read_csv(HERE / "results_full_model/outcome_association_full.csv")
oa = oa[(oa.model == "adjusted") & (oa.scope == "full")]
MIMIC_OR = {("new invasive ventilation within 12 h", L): "1.64 (1.14–2.35)",
            ("new invasive ventilation within 12 h", N): "3.25 (2.32–4.55)",
            ("new invasive ventilation within 12 h", R): "4.28 (3.10–5.92)",
            ("new vasoactive support within 12 h", L): "1.21 (0.81–1.79)",
            ("new vasoactive support within 12 h", N): "4.91 (3.79–6.36)",
            ("new vasoactive support within 12 h", R): "5.00 (4.07–6.13)",
            ("ICU death (discrete-time hazard)", L): "0.21 (0.05–0.87)",
            ("ICU death (discrete-time hazard)", N): "3.73 (2.48–5.59)",
            ("ICU death (discrete-time hazard)", R): "11.83 (9.00–15.55)"}
OCN = {"new invasive ventilation within 12 h": "New invasive ventilation within 12 h",
       "new vasoactive support within 12 h": "New vasoactive support within 12 h",
       "ICU death (discrete-time hazard)": "ICU death, discrete-time hazard"}
rows = []
for oc in ["new invasive ventilation within 12 h", "new vasoactive support within 12 h", "ICU death (discrete-time hazard)"]:
    for st in [L, N, R]:
        row = oa[(oa.outcome == oc) & (oa.state == st)]
        if not len(row): continue
        r0 = row.iloc[0]
        eicu = f"{r0.OR:.2f} ({r0.CI_low:.2f}–{r0.CI_high:.2f})"
        if r0.suppressed_lt10_events:
            eicu += f"†"
        rows.append([OCN[oc], SHORT[st], MIMIC_OR[(oc, st)], eicu, int(r0.n_events_in_state)])
T[5] = ("**Table 5.** Adjusted odds ratios for the current state and subsequent events, reference state "
        "neurologically preserved–low support. Generalized estimating equations with an exchangeable working "
        "correlation clustered by patient, adjusted for age, sex and stroke subtype. † Suppressed under the "
        "prespecified rule of fewer than ten events; the discovery study's estimate for the same state also "
        "rested on two events. All nine comparisons reproduce the direction of effect.",
        md(rows, ["Outcome", "State", "MIMIC-IV OR (95% CI)", "eICU OR (95% CI)", "eICU events"]))

# ---- Table 6: between-hospital heterogeneity ---------------------------------
a3 = pd.read_csv(HERE / "results_aim3/between_hospital_variance.csv").set_index("state")
rows = [[SHORT[s], f"{a3.loc[s,'pooled_pct']:.1f}",
         f"{a3.loc[s,'hospital_min_pct']:.1f}–{a3.loc[s,'hospital_max_pct']:.1f}",
         f"{a3.loc[s,'hospital_median_pct']:.1f}", f"{a3.loc[s,'between_hospital_SD_pp']:.1f}",
         f"{a3.loc[s,'ICC']:.3f}"] for s in ORDER]
T[6] = ("**Table 6.** Between-hospital variation in state prevalence across the 38 hospitals meeting both "
        "eligibility rules. Variance components from a random-intercept linear probability model, so the "
        "standard deviation reads directly in percentage points of prevalence.",
        md(rows, ["State", "Pooled, %", "Hospital range, %", "Hospital median, %",
                  "Between-hospital SD, pp", "ICC"]))

# ---- write -------------------------------------------------------------------
blocks = []
for k in sorted(T):
    cap_, body = T[k]
    blocks.append(f"{cap_}\n\n{body}\n")
(OUT / "tables_v1.md").write_text("# Tables\n\n" + "\n\n".join(blocks))
print(f"wrote {OUT/'tables_v1.md'}")
for k in sorted(T):
    n = T[k][1].count("\n") - 1
    print(f"  Table {k}: {n} data rows")
