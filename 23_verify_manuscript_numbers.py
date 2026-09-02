"""
Paper 2: verify every headline number in the manuscript against its source CSV.

Mirrors 03_code/15_verify_manuscript_numbers.py. A number that appears in the
prose and cannot be re-derived from a pipeline output is a defect, whether it is
wrong or merely untraceable.
"""
import sys
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
# Two manuscripts are maintained: the full version, and the 4000-word JAMIA
# version derived from it. Both must survive the same checks, because a number
# that survives compression unchanged is the only kind worth compressing.
FILES = {
    "full": ["abstract_draft_v1.md", "introduction_draft_v1.md", "methods_draft_v1.md",
             "results_draft_v1.md", "discussion_draft_v1.md"],
    "jamia": ["abstract_jamia_EN.md", "intro_jamia_EN.md", "methods_jamia_EN.md",
              "results_jamia_EN.md", "discussion_jamia_EN.md"],
}
VARIANT = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "full"
assert VARIANT in FILES, f"unknown variant {VARIANT!r}; choose from {list(FILES)}"
M = " ".join((HERE / "manuscript" / f).read_text() for f in FILES[VARIANT])
print(f"verifying the {VARIANT} manuscript\n")

flow = pd.read_csv(HERE / "cohort/cohort_flow_counts.csv")
coh = pd.read_csv(HERE / "cohort/patient_level_cohort.csv")
tm = pd.read_csv(HERE / "results_full_model/transport_metrics_full.csv").set_index("scope")
ps = pd.read_csv(HERE / "results_full_model/per_state_full.csv")
psb = ps[ps.scope == "both"].set_index("state")
tf = pd.read_csv(HERE / "results_treatment_free/per_state_transport.csv").set_index("state")
av = pd.read_csv(HERE / "audit/availability_vs_mimic.csv").set_index("variable")
a3 = pd.read_csv(HERE / "results_aim3/between_hospital_variance.csv").set_index("state")
dn = pd.read_csv(HERE / "results_denovo/denovo_state_matching.csv")
ks = pd.read_csv(HERE / "results_denovo/denovo_K_selection.csv").set_index("K")
oa = pd.read_csv(HERE / "results_full_model/outcome_association_full.csv")
og = pd.read_csv(HERE / "results_full_model/outcome_gradient_full.csv").set_index("state")
cap = pd.read_csv(HERE / "review_returned/ctier_capture_check.csv").set_index("variable")
k3 = pd.read_csv(HERE / "results_denovo/denovo_K3_matching.csv")
k3s = k3.set_index('best_match_state')
k3p = k3.set_index('eicu_state').pct_windows
k4 = pd.read_csv(HERE / 'results_denovo/denovo_K4_matching.csv')
cm4 = pd.read_csv(HERE / 'results_denovo/denovo_K4_full_corr_matrix.csv', index_col=0)
gc = pd.read_csv(HERE / "results_full_model/gcs_observed_only_profile_corr.csv")
gs = pd.read_csv(HERE / "results_full_model/gcs_observed_vs_imputed_prevalence.csv")
am = pd.read_csv(HERE / "manuscript/supplementary_amendment_log.csv")
el = pd.read_csv(HERE / "cohort/patient_level_cohort_eligible.csv")

P, R, L = "neurologically preserved-low support", "neurological impairment-respiratory support", "neurological impairment-low support"
N = "neurological impairment-renal dysfunction"
oaf = oa[(oa.model == "adjusted") & (oa.scope == "full")].set_index(["outcome", "state"])
MV, VP, DE = "new invasive ventilation within 12 h", "new vasoactive support within 12 h", "ICU death (discrete-time hazard)"

CHECKS = [
    ("cohort patients", "8279", str(int(flow.n_patients.iloc[-1]))),
    ("cohort hospitals", "176", str(int(flow.n_hospitals.iloc[-1]))),
    ("total windows", "70630", str(int(coh.n_windows.sum()))),
    ("phenotype stays", "11023", str(int(flow.n_stays.iloc[0]))),
    ("after trauma exclusion", "10644", str(int(flow.n_stays.iloc[1]))),
    ("first unit stay", "8995", str(int(flow.n_stays.iloc[3]))),
    ("LOS>=12h", "8522", str(int(flow.n_stays.iloc[4]))),
    ("median age", "68", f"{coh.age_n.median():.0f}"),
    ("female %", "48.5", f"{(coh.gender=='Female').mean()*100:.1f}"),
    ("ICU mortality %", "7.3", f"{coh.icu_mortality.mean()*100:.1f}"),
    ("hospital mortality %", "13.2", f"{coh.hospital_mortality.mean()*100:.1f}"),
    ("neuro ICU n", "2884", str(int((coh.unittype=='Neuro ICU').sum()))),
    ("GCS-eligible patients", "4568", str(len(el))),
    ("vent-ascertainable patients", "7500", str(int(coh.vent_ascertainable.sum()))),
    ("both-scope patients", "4296", str(int(tm.loc['both','patients']))),
    ("both-scope windows", "37070", str(int(tm.loc['both','windows']))),
    ("min profile corr, both", "0.912", f"{tm.loc['both','min_profile_corr']:.3f}"),
    ("transition rank corr, both", "0.974", f"{tm.loc['both','transition_rank_corr']:.3f}"),
    ("max self-transition diff", "0.039", f"{tm.loc['both','max_self_transition_diff']:.3f}"),
    ("pct changing state", "38.1", f"{tm.loc['both','pct_changing_state']:.1f}"),
    ("preserved prev, both", "67.0", f"{psb.loc[P,'eicu_prev']:.1f}"),
    ("preserved corr, both", "0.970", f"{psb.loc[P,'profile_corr']:.3f}"),
    ("respiratory prev, both", "13.5", f"{psb.loc[R,'eicu_prev']:.1f}"),
    ("low-support corr, both", "0.912", f"{psb.loc[L,'profile_corr']:.3f}"),
    ("GCS availability eICU", "53.5", f"{av.loc['gcs_eye','eicu_pct_windows']:.1f}"),
    ("GCS availability MIMIC", "94.9", f"{av.loc['gcs_eye','mimic_pct_windows']:.1f}"),
    ("urine availability eICU", "55.6", f"{av.loc['urine_output_ml','eicu_pct_windows']:.1f}"),
    ("sedation capture", "13.4", f"{cap.loc['sedative','eicu_pct_windows']:.1f}"),
    ("vasoactive capture", "6.8", f"{cap.loc['vasopressor','eicu_pct_windows']:.1f}"),
    ("mech vent lower bound", "17.2", f"{cap.loc['mech_vent','eicu_pct_windows']:.1f}"),
    ("TF low-support corr", "0.709", f"{tf.loc[L,'corr_eligible']:.3f}"),
    ("TF respiratory corr", "0.979", f"{tf.loc[R,'corr_eligible']:.3f}"),
    ("ICC preserved", "0.026", f"{a3.loc[P,'ICC']:.3f}"),
    ("ICC low-support", "0.012", f"{a3.loc[L,'ICC']:.3f}"),
    ("between-hosp SD preserved", "7.6", f"{a3.loc[P,'between_hospital_SD_pp']:.1f}"),
    ("denovo K4 BIC", "754215", f"{ks.loc[4,'BIC']:.0f}"),
    ("denovo K3 BIC", "826358", f"{ks.loc[3,'BIC']:.0f}"),
    ("denovo K4 stability", "0.840", f"{ks.loc[4,'restart_stability_ARI']:.3f}"),
    ("denovo unmatched corr", "0.027", f"{dn.loc[~dn.corresponds,'profile_correlation'].iloc[0]:.3f}"),
    ("denovo respiratory corr", "0.979", f"{dn.loc[dn.best_mimic_candidate==R,'profile_correlation'].iloc[0]:.3f}"),
    ("MV OR respiratory", "7.52", f"{oaf.loc[(MV,R),'OR']:.2f}"),
    ("MV OR renal", "4.46", f"{oaf.loc[(MV,N),'OR']:.2f}"),
    ("death OR respiratory", "44.51", f"{oaf.loc[(DE,R),'OR']:.2f}"),
    ("death OR renal", "9.91", f"{oaf.loc[(DE,N),'OR']:.2f}"),
    ("VP OR renal", "3.60", f"{oaf.loc[(VP,N),'OR']:.2f}"),
    ("last-state mortality preserved", "2.2", f"{og.loc[P,'icu_death_pct']:.1f}"),
    ("last-state mortality respiratory", "41.6", f"{og.loc[R,'icu_death_pct']:.1f}"),
    ("last-state mortality renal", "18.7", f"{og.loc[N,'icu_death_pct']:.1f}"),
    # --- revision v2 additions ---
    ("K3 stability ARI", "0.991", f"{ks.loc[3,'restart_stability_ARI']:.3f}"),
    ("K3 preserved corr", "0.988", f"{k3s.loc[P,'best_match_correlation']:.3f}"),
    ("K3 renal corr", "0.991", f"{k3s.loc[N,'best_match_correlation']:.3f}"),
    ("K3 respiratory corr", "0.971", f"{k3s.loc[R,'best_match_correlation']:.3f}"),
    ("K3 preserved prev", "68.1", f"{k3p[1]:.1f}"),
    ("K3 respiratory prev", "17.4", f"{k3p[0]:.1f}"),
    ("K3 renal prev", "14.5", f"{k3p[2]:.1f}"),
    # --- de novo K=4 correction ---
    ("K4 state2 best match corr", "0.680", f"{k4.set_index('eicu_state').loc[2,'best_match_correlation']:.3f}"),
    ("K4 state3 best match corr", "0.962", f"{k4.set_index('eicu_state').loc[3,'best_match_correlation']:.3f}"),
    ("K4 state2 pct in frozen preserved", "81.4", f"{k4.set_index('eicu_state').loc[2,'pct_in_modal_frozen_state']:.1f}"),
    ("K4 state3 pct in frozen preserved", "99.9", f"{k4.set_index('eicu_state').loc[3,'pct_in_modal_frozen_state']:.1f}"),
    ("K4 two preserved states combined", "73.1",
     f"{k4.set_index('eicu_state').loc[2,'pct_windows']+k4.set_index('eicu_state').loc[3,'pct_windows']:.1f}"),
    ("impairment-low highest denovo corr", "0.183", f"{cm4[L].max():.3f}"),
    ("obs-GCS min corr, full", "0.944", f"{gc[gc.scope=='full'].corr_observed_gcs_only.min():.3f}"),
    ("obs-GCS min corr, both", "0.938", f"{gc[gc.scope=='both-eligible'].corr_observed_gcs_only.min():.3f}"),
    ("preserved obs-GCS pct, full", "66.1", f"{gs[(gs.scope=='full')&(gs.state==P)].eicu_observed_gcs_pct.iloc[0]:.1f}"),
    ("preserved imputed-GCS pct, full", "82.0", f"{gs[(gs.scope=='full')&(gs.state==P)].eicu_imputed_gcs_pct.iloc[0]:.1f}"),
    ("preserved obs-GCS pct, both", "66.0", f"{gs[(gs.scope=='both-eligible')&(gs.state==P)].eicu_observed_gcs_pct.iloc[0]:.1f}"),
    ("preserved imputed-GCS pct, both", "70.9", f"{gs[(gs.scope=='both-eligible')&(gs.state==P)].eicu_imputed_gcs_pct.iloc[0]:.1f}"),
    ("respiratory obs-GCS pct, full", "13.8", f"{gs[(gs.scope=='full')&(gs.state==R)].eicu_observed_gcs_pct.iloc[0]:.1f}"),
    ("amendments logged", "fifteen", "fifteen" if len(am)==15 else str(len(am))),
]

# ---- Aim 4: prediction ladder, horizons and contrasts ------------------------
pm = pd.read_csv(HERE / "prediction/metrics_by_model.csv")
pq = pd.read_csv(HERE / "prediction/dataset_qa.csv").set_index("scope")
ph = pd.read_csv(HERE / "prediction/multihorizon_metrics.csv")
pc = pd.read_csv(HERE / "prediction/horizon_contrasts.csv")
ps = pd.read_csv(HERE / "prediction/horizon_sensitivity.csv")
ef = pm[pm.scope == "eicu-full"].set_index("model")
mt = pm[pm.scope == "mimic-test"].set_index("model")
hf = ph[ph.scope == "eicu-full"].set_index(["horizon_h", "model"])
cf = pc[pc.scope == "eicu-full"].set_index(["horizon_h", "contrast"])
fx = ps[ps.analysis == "population fixed at 24 h"].set_index(["scope", "horizon_h"])
l6 = pd.read_csv(HERE / "prediction/metrics_L6.csv").set_index(["scope", "horizon_h"])
sq = pd.read_csv(HERE / "prediction/sequence_contrasts.csv").set_index(["scope", "horizon_h"])
_sm = pd.read_csv(HERE / "prediction/sequence_model_metrics.csv")
seed_spread = float((_sm.groupby(["scope", "horizon_h"]).L6_auroc.max()
                     - _sm.groupby(["scope", "horizon_h"]).L6_auroc.min()).max())

def share(scope, h):
    a = hf.loc[(h, "L2"), "change_auroc"] if scope == "eicu-full" else \
        ph[(ph.scope == scope) & (ph.horizon_h == h) & (ph.model == "L2")].change_auroc.iloc[0]
    return (a - 0.5) / (l6.loc[(scope, h), "change_auroc"] - 0.5) * 100

CHECKS += [
    ("persistence, eICU", "90.7", f"{pq.loc['eicu-full','persistence_pct']:.1f}"),
    ("persistence, MIMIC test", "91.4", f"{pq.loc['mimic-test','persistence_pct']:.1f}"),
    ("changed pct, eICU 6h", "9.3", f"{pq.loc['eicu-full','changed_pct']:.1f}"),
    ("pairs, eICU 6h", "62,351", f"{int(pq.loc['eicu-full','pairs_4class']):,}"),
    ("L1 AUROC eICU", "0.684", f"{ef.loc['L1','change_auroc']:.3f}"),
    ("L2 AUROC eICU", "0.730", f"{ef.loc['L2','change_auroc']:.3f}"),
    ("L5 AUROC eICU", "0.766", f"{ef.loc['L5','change_auroc']:.3f}"),
    ("L2 AUROC MIMIC test", "0.735", f"{mt.loc['L2','change_auroc']:.3f}"),
    ("L1 AUROC MIMIC test", "0.694", f"{mt.loc['L1','change_auroc']:.3f}"),
    ("L2 sens at 90 spec", "27.8", f"{ef.loc['L2','change_sens_at_90spec']*100:.1f}"),
    ("L5 sens at 90 spec", "36.5", f"{ef.loc['L5','change_sens_at_90spec']*100:.1f}"),
    ("L2 AUROC eICU 12h", "0.754", f"{hf.loc[(12,'L2'),'change_auroc']:.3f}"),
    ("L2 AUROC eICU 24h", "0.780", f"{hf.loc[(24,'L2'),'change_auroc']:.3f}"),
    ("pairs eICU 12h", "54,072", f"{int(hf.loc[(12,'L2'),'pairs']):,}"),
    ("pairs eICU 24h", "38,011", f"{int(hf.loc[(24,'L2'),'pairs']):,}"),
    ("changed pct eICU 12h", "15.2", f"{hf.loc[(12,'L2'),'changed_pct']:.1f}"),
    ("changed pct eICU 24h", "19.1", f"{hf.loc[(24,'L2'),'changed_pct']:.1f}"),
    ("L2-L0 eICU 24h", "0.281", f"{cf.loc[(24,'L2 - L0'),'delta_auroc']:.3f}"),
    ("L5-L2 eICU 24h", "0.005", f"{cf.loc[(24,'L5 - L2'),'delta_auroc']:.3f}"),
    ("L5-L2 eICU 24h CI lower", "\u22120.001", f"{cf.loc[(24,'L5 - L2'),'lo']:.3f}".replace("-", "\u2212")),  # prose uses U+2212
    ("L5-L2 eICU 24h CI upper", "0.011", f"{cf.loc[(24,'L5 - L2'),'hi']:.3f}"),
    ("fixed-population L2 6h", "0.712", f"{fx.loc[('eicu-full',6),'L2']:.3f}"),
    ("fixed-population L2 24h", "0.780", f"{fx.loc[('eicu-full',24),'L2']:.3f}"),
    # --- sequence-model ceiling (step 38/39) ---
    ("L6 AUROC eICU 6h", "0.787", f"{l6.loc[('eicu-full',6),'change_auroc']:.3f}"),
    ("L6 AUROC eICU 24h", "0.798", f"{l6.loc[('eicu-full',24),'change_auroc']:.3f}"),
    ("L6-L2 eICU 6h", "0.057", f"{sq.loc[('eicu-full',6),'delta_auroc']:.3f}"),
    ("L6-L2 eICU 24h", "0.018", f"{sq.loc[('eicu-full',24),'delta_auroc']:.3f}"),
    ("frozen share eICU 6h", "80.2", f"{share('eicu-full',6):.1f}"),
    ("frozen share eICU 12h", "84.9", f"{share('eicu-full',12):.1f}"),
    ("frozen share eICU 24h", "94.1", f"{share('eicu-full',24):.1f}"),
    ("frozen share MIMIC 6h", "82.8", f"{share('mimic-test',6):.1f}"),
    ("frozen share MIMIC 24h", "89.6", f"{share('mimic-test',24):.1f}"),
    ("seed spread, worst case", "0.009", f"{seed_spread:.3f}"),

    # --- v4 polish additions ---
    ("preserved hospital min", "46.6", f"{a3.loc[P,'hospital_min_pct']:.1f}"),
    ("preserved hospital max", "86.3", f"{a3.loc[P,'hospital_max_pct']:.1f}"),
    ("renal hospital min", "2.1", f"{a3.loc[N,'hospital_min_pct']:.1f}"),
    ("renal hospital max", "25.6", f"{a3.loc[N,'hospital_max_pct']:.1f}"),
]
bad = []
print(f"{'claim':<36}{'in manuscript':>15}{'from CSV':>14}   status")
print("-" * 82)
for name, claimed, actual in CHECKS:
    in_text = claimed in M or f"{int(claimed):,}" in M if claimed.isdigit() else claimed in M
    match = claimed == actual
    st = "OK" if (match and in_text) else ("MISMATCH" if not match else "not found in text")
    if st != "OK":
        bad.append((name, claimed, actual, st))
    print(f"{name:<36}{claimed:>15}{actual:>14}   {st}")
print("-" * 82)
print(f"{len(CHECKS) - len(bad)} of {len(CHECKS)} verified")
if bad:
    print("\nPROBLEMS:")
    for n, c, a, s in bad:
        print(f"  {n}: manuscript says {c}, CSV says {a} [{s}]")
pd.DataFrame([{"claim": n, "manuscript": c, "source_csv": a,
               "status": "OK" if (c == a) else "MISMATCH"} for n, c, a in CHECKS]).to_csv(
    HERE / "manuscript/numbers_audit.csv", index=False)
