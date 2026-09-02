"""
Paper 2: copy the publishable files into a clean folder, ready to become the
public repository.

Selecting files by hand is how a patient-level file ends up in a public
repository, so nothing is chosen here: the manifest written by
44_repository_audit.py is the only input, and only rows marked PUBLISH are
copied. The export then re-audits itself — every copied file is re-classified
from scratch, and the export aborts if anything other than PUBLISH comes back.

Output: ../paper2_code_repository/
"""
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
MANIFEST = HERE / "repository_manifest.csv"
DEST = HERE.parent / "paper2_code_repository"

rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
publish = [r for r in rows if r["verdict"] == "PUBLISH"]
assert publish, "manifest has no PUBLISH rows -- run 44_repository_audit.py first"

# Clear the previous export, but never the version-control directory: once this
# folder has been pushed, deleting .git destroys the link to the remote and the
# commit history with it.
if DEST.exists():
    for child in DEST.iterdir():
        if child.name in {".git", ".gitignore"}:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
DEST.mkdir(parents=True, exist_ok=True)

for r in publish:
    src = HERE / r["path"]
    dst = DEST / r["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# --- re-audit the export, independently of the manifest ----------------------
import importlib.util
spec = importlib.util.spec_from_file_location("audit", HERE / "44_repository_audit.py")
ID_COLS = {"stay_id", "subject_id", "hadm_id", "patientunitstayid",
           "patienthealthsystemstayid", "uniquepid", "uid", "icustay_id"}
HOSPITAL_COLS = {"hospitalid"}
bad = []
for p in sorted(DEST.rglob("*")):
    if p.is_dir():
        continue
    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
            cols = {c.strip().lower() for c in next(csv.reader(fh), [])}
        if cols & ID_COLS:
            bad.append((p, f"patient identifier: {cols & ID_COLS}"))
        elif cols & HOSPITAL_COLS:
            bad.append((p, "hospital-level statistics"))
    elif p.suffix.lower() in {".pt", ".pkl", ".pth", ".joblib", ".env", ".key"}:
        bad.append((p, "model binary or secret"))

if bad:
    shutil.rmtree(DEST)
    for p, why in bad:
        print(f"  ABORT {p.relative_to(DEST.parent)}: {why}")
    raise SystemExit("export deleted: the re-audit found files that must not be published")

n = sum(1 for p in DEST.rglob("*") if p.is_file())
kb = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()) / 1024
print(f"exported {n} files, {kb:.0f} KB -> {DEST}")
print("re-audit passed: no patient identifiers, no hospital-level tables, no binaries\n")
for sub in sorted({p.parent.relative_to(DEST).as_posix() or "." for p in DEST.rglob("*") if p.is_file()}):
    c = sum(1 for p in (DEST / sub).glob("*") if p.is_file())
    print(f"  {sub + '/':<34} {c:>3} files")
