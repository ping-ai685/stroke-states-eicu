"""
Paper 2: decide, file by file, what goes into the public code repository.

Both source databases are governed by the PhysioNet Credentialed Health Data Use
Agreement, which forbids redistribution of the data and of patient-level material
derived from it. Publishing one such file by mistake breaches that agreement, so
files are classified by inspecting content, not by trusting extensions.

Three tiers:

  EXCLUDE   must not be published: a patient identifier column, a model binary,
            a credential, or hospital-level derived statistics. Hospital-level
            tables are excluded even though they carry no patient identifier,
            because they are derived data at fine granularity and the code
            regenerates them from source for anyone who has access.
  OMIT      safe to publish but left out: regenerable result tables, manuscript
            drafts, figures, and the frozen model parameters, which the Data
            Availability Statement says are supplied on request to credentialed
            users rather than posted.
  PUBLISH   the analysis code, the frozen term dictionaries and phenotype paths
            the article asks readers to reuse, and the three supplementary
            tables that appear in the article anyway.

A row-count heuristic was tried first and wrongly excluded the drug term list,
which has 499 rows because there are 499 drug names. Counts of rows say nothing
about whose rows they are; only the columns do.

Output: repository_manifest.csv
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "repository_manifest.csv"

ID_COLS = {"stay_id", "subject_id", "hadm_id", "patientunitstayid",
           "patienthealthsystemstayid", "uniquepid", "uid", "icustay_id"}
HOSPITAL_COLS = {"hospitalid"}
NEVER_EXT = {".pt", ".pkl", ".pth", ".joblib", ".env", ".key", ".pem"}
CODE_EXT = {".py"}

PUBLISH_EXACT = {
    "README.md",
    "frozen_dictionary_v1.0/MANIFEST.csv",
    "frozen_dictionary_v1.0/terms_drugs.csv",
    "frozen_dictionary_v1.0/terms_urine_sources.csv",
    "frozen_dictionary_v1.0/terms_ventilation_crrt.csv",
    "frozen_phenotype/phenotype_paths_frozen.csv",
    "harmonization_dictionary.csv",
    "harmonization_dictionary_cohort_fields.csv",
    "manuscript/supplementary_amendment_log.csv",
}


def classify(p: Path):
    rel = p.relative_to(HERE).as_posix()
    ext = p.suffix.lower()
    if p.name.startswith("."):
        return "EXCLUDE", "hidden file"
    if ext in NEVER_EXT:
        return "EXCLUDE", f"model binary or secret ({ext})"
    if ext in CODE_EXT:
        return "PUBLISH", "analysis code"
    if ext == ".csv":
        try:
            with p.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
                header = next(csv.reader(fh), [])
        except Exception as e:
            return "EXCLUDE", f"unreadable, excluded by default: {e}"
        cols = {c.strip().lower() for c in header}
        hit = cols & ID_COLS
        if hit:
            return "EXCLUDE", f"patient identifier column: {', '.join(sorted(hit))}"
        if cols & HOSPITAL_COLS:
            return "EXCLUDE", "hospital-level derived statistics"
        if rel in PUBLISH_EXACT:
            return "PUBLISH", "frozen dictionary or supplementary table"
        return "OMIT", "regenerable result table"
    if rel.startswith("frozen_params"):
        return "OMIT", "frozen parameters: supplied on request, per the Data Availability Statement"
    if ext in {".png", ".tif", ".tiff"}:
        return "OMIT", "figure, published with the article"
    if rel in PUBLISH_EXACT:
        return "PUBLISH", "repository documentation"
    if ext in {".docx", ".md", ".log", ".txt"}:
        return "OMIT", "manuscript draft or working note"
    return "EXCLUDE", f"unrecognised type {ext}, excluded by default"


rows = []
for p in sorted(HERE.rglob("*")):
    if p.is_dir() or p.name == OUT.name:
        continue
    v, why = classify(p)
    rows.append({"path": p.relative_to(HERE).as_posix(), "verdict": v,
                 "reason": why, "kb": round(p.stat().st_size / 1024, 1)})

with OUT.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["path", "verdict", "reason", "kb"])
    w.writeheader(); w.writerows(rows)

for v in ("PUBLISH", "OMIT", "EXCLUDE"):
    sel = [r for r in rows if r["verdict"] == v]
    print(f"{v:<9} {len(sel):>4} files  {sum(r[chr(107)+chr(98)] for r in sel)/1024:>7.2f} MB")
print(f"\nwrote {OUT}\n")
print("PUBLISH set:")
for r in rows:
    if r["verdict"] == "PUBLISH" and not r["path"].endswith(".py"):
        print(f"  {r['path']:<52} {r['reason']}")
npy = sum(1 for r in rows if r["verdict"] == "PUBLISH" and r["path"].endswith(".py"))
print(f"  + {npy} .py analysis scripts")
print("\nEXCLUDE, by reason:")
import collections
for why, n in collections.Counter(r["reason"].split(":")[0] for r in rows
                                  if r["verdict"] == "EXCLUDE").most_common():
    print(f"  {n:>3}  {why}")
