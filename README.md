# Transportability and independent rediscovery of dynamic clinical states in acute stroke

Analysis code for the study of the same name, in which a four-state hidden Markov
representation of the first 72 h of intensive care after acute stroke — derived in
MIMIC-IV and reported separately [1] — is applied unchanged to 8,279 patients
across 176 hospitals in the eICU Collaborative Research Database, and then set
aside so that a model can be fitted to eICU from scratch.

The study asks three questions that are usually reported as one:

1. **Does the model transport?** Applied frozen, does it assign states that behave
   as the original states did?
2. **Do the transported states carry forward-looking information?** Does the state
   a patient occupies now say anything about the state they occupy next?
3. **Is the structure independently rediscoverable?** Given the same variables and
   no knowledge of the original solution, does eICU arrive at it?

The answers were not the same, which is the point of the paper.

---

## This repository contains no patient data, and cannot

MIMIC-IV and eICU-CRD are held at PhysioNet under credentialed access and are
governed by the PhysioNet Credentialed Health Data Use Agreement 1.5.0, which
forbids redistribution of the data and of patient-level material derived from it.
Nothing here is a substitute for obtaining the databases yourself:

- MIMIC-IV v3.1 — https://physionet.org/content/mimiciv/
- eICU-CRD v2.0 — https://physionet.org/content/eicu-crd/

Access is open to any investigator who becomes a credentialed PhysioNet user,
completes the CITI *Data or Specimens Only Research* training, and signs the
agreement.

Every file in this repository was classified before publication by
`44_repository_audit.py`, which inspects the columns of each tabular file rather
than trusting its name or size, and excludes anything carrying a patient
identifier or hospital-level derived statistics. The resulting decisions are
recorded in `repository_manifest.csv`.

**The frozen model parameters are not posted here.** They are aggregate
quantities containing no patient-level records, but they are derived from
credentialed data, so they are supplied by the corresponding author to any
investigator who holds credentialed access to MIMIC-IV. With them the transport
can be reproduced exactly rather than refitted as an approximation.

---

## What is here

### The frozen dictionaries the article asks you to reuse

These are the part of the work most likely to be useful to someone else
transporting a model between MIMIC-IV and eICU-CRD, and they are frozen artefacts
rather than intermediate output:

| File | Contents |
|---|---|
| `frozen_phenotype/phenotype_paths_frozen.csv` | the 35 enumerated eICU `diagnosisstring` paths of the stroke phenotype, with subtype and include/exclude decision |
| `harmonization_dictionary.csv` | all 21 model variables mapped to eICU source table, field, selector, units, offset field and aggregation rule |
| `frozen_dictionary_v1.0/terms_*.csv` | the clinically adjudicated free-text term lists for the four organ-support variables |
| `manuscript/supplementary_amendment_log.csv` | every protocol amendment, dated and classified, with what was visible at the time |

Three findings in these files are worth stating plainly, because they cost time to
discover: in eICU, `icd9code` is a deterministic re-encoding of `diagnosisstring`
and therefore carries no independent phenotyping information; invasive ventilation
ascertainment is a property of the hospital rather than of the patient; and a
fitted emission with near-zero variance can make an otherwise reasonable
sensitivity analysis non-executable.

### The analysis, in run order

Scripts are numbered in dependency order and each writes its output to a
subdirectory that the next one reads.

| Step | Scripts | What happens |
|---|---|---|
| Freeze the discovery model | `00` | export every MIMIC-IV parameter the transport will need, and assert that replaying them reproduces the published assignments exactly |
| Phenotype and cohort | `01`–`03` | enumerate and freeze the stroke phenotype, then build the analysis cohort |
| Measurement audit | `04`–`06`, `17` | hospital-level interface availability; the two eligibility rules |
| Numerical guards | `07`–`08` | verify the epsilon floor changes nothing; diagnose the GCS point-mass emission |
| Windows and preprocessing | `09`–`10` | build the 6-hour window table and apply the frozen preprocessing |
| Term dictionaries | `14`–`16`, `18` | generate term lists for clinical review, validate what comes back, freeze |
| Transport | `11`–`13`, `19`–`20` | treatment-free control first, then the full 21-variable model, then outcomes |
| Heterogeneity | `22` | between-hospital variance components |
| Independent refitting | `21`, `24` | fit eICU from scratch and compare with the discovery states |
| Prediction | `29`–`32`, `34`–`36`, `38`–`39` | the predictor ladder, three horizons, the sequence-model ceiling |
| Verification | `23` | re-derive every headline number in the manuscript from its source file |
| Tables and figures | `25`–`28`, `33`, `37`, `40`–`43` | everything reported in the article |
| Repository audit | `44` | the classification that produced this repository |

`figure_labels.py` holds the display text for the figures in both languages, so
the English and Chinese sets are drawn by the same plotting code.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn statsmodels matplotlib torch pomegranate
```

Set the paths to your own copies of the two databases at the top of `02` and `03`,
then run the scripts in numerical order. Several take tens of minutes; the
sequence model (`38`) trains three seeds at each of three horizons.

Two conventions are worth knowing before reading the code:

- **Nothing is estimated from eICU during state assignment.** Emission and
  transition parameters, imputation constants, scaling constants and state labels
  are loaded from `frozen_params/` and never recomputed. Scripts assert this.
- **Scripts verify themselves.** `23` re-derives every number in the manuscript
  from its source file; `28` checks each extracted table against the manuscript
  cell by cell; `31` asserts that the persistence null scores exactly 0.5, which
  is what caught a state-label misalignment during development.

---

## Citation

If this code is useful, please cite the article, and the discovery study whose
model it transports:

> [1] Lei P, Xu Y, Zhang Y. Dynamic clinical states and transitions during the
> first 72 hours of intensive care after acute stroke. *medRxiv*. 2026.
> doi:10.64898/2026.08.30.26361738

## Licence

Code is released under the MIT Licence. The frozen dictionaries are released
under CC BY 4.0. Neither covers the underlying databases, which remain subject to
the PhysioNet agreement.

## Contact

Ping Lei — Ping.Pinglei@unige.ch — ORCID 0000-0002-8904-3167
