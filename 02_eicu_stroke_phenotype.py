"""
Paper 2, step 2: the FROZEN eICU acute-stroke phenotype.

Protocol v1.0 amendment A, approved by the PI on 23 August 2026:
  - the primary phenotype is the `diagnosis.diagnosisstring` path, NOT Paper 1's
    ICD code list. eICU's `icd9code` is a deterministic re-encoding of the same
    string, so the two are one source, not two; and Paper 1's list would drop
    eICU's generic "stroke" label (ICD-9 436), which Paper 1 deliberately
    excluded as ill-defined but which eICU applies to 3,897 stays.
  - stays carrying only the unspecified `…|stroke` path ARE included in the
    primary cohort. A sensitivity analysis excluding them is pre-specified.
  - `patient.apacheadmissiondx` is the one genuinely independent cross-check.

Paths are enumerated explicitly rather than matched by prefix. That is not
pedantry: `…|hemorrhagic stroke|subarachnoid hemorrhage|traumatic` sits INSIDE
the stroke branch, so a prefix match would admit 48 traumatic SAH stays that the
`trauma - CNS` exclusion never sees. Enumerating forces every path to be looked
at once.

Subtype precedence replicates Paper 1's `subtype_label` exactly: a stay's
subtypes are collected as a set, ICH+SAH together is labelled "ICH+SAH", and
otherwise AIS beats ICH beats SAH.

Outputs (frozen_phenotype/):
  phenotype_paths_frozen.csv    every path, its subtype contribution, its status
  eicu_stroke_stays.csv         stay-level phenotype and subtype
  phenotype_crosscheck.csv      agreement with apacheadmissiondx
"""
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import data_paths                                    # noqa: E402

B = str(data_paths.eicu())
OUT = Path(__file__).parent / "frozen_phenotype"
OUT.mkdir(exist_ok=True)
FREEZE_DATE = "2026-08-23"
P = "neurologic|disorders of vasculature|stroke"

# ---------------------------------------------------------------------------
# The frozen path table. `subtypes` is what the path contributes; `status` is
# include / exclude / review. "review" paths are INCLUDED in the primary cohort
# and carry a flag so the pre-specified sensitivity analysis can drop them.
# ---------------------------------------------------------------------------
PATHS = [
    # ---- generic (the amendment-A decision) ----
    ("",                                                      ("unspecified",), "include", "generic stroke label; eICU codes it 436/I67.8"),
    # ---- ischemic ----
    ("|ischemic stroke",                                      ("AIS",), "include", ""),
    ("|ischemic stroke|left sided",                           ("AIS",), "include", ""),
    ("|ischemic stroke|right sided",                          ("AIS",), "include", ""),
    ("|ischemic stroke|cardioembolic",                        ("AIS",), "include", ""),
    ("|ischemic stroke|atherosclerotic thrombosis",           ("AIS",), "include", ""),
    ("|ischemic stroke|cortical",                             ("AIS",), "include", ""),
    ("|ischemic stroke|atheroembolic",                        ("AIS",), "include", ""),
    ("|ischemic stroke|dissection of cervicocerebral arteries",("AIS",), "include", ""),
    ("|ischemic stroke|brainstem",                            ("AIS",), "include", ""),
    ("|ischemic stroke|lacunar infarct",                      ("AIS",), "include", ""),
    ("|ischemic stroke|from cortical venous thrombosis",      ("AIS",), "include", ""),
    ("|ischemic stroke|vasospasm, including cocaine-induced", ("AIS",), "include", ""),
    ("|ischemic stroke|perioperative",                        ("AIS",), "review",  "peri-procedural rather than spontaneous stroke"),
    # ---- haemorrhagic (ICH) ----
    ("|hemorrhagic stroke",                                   ("ICH",), "include", ""),
    ("|hemorrhagic stroke|hypertensive",                      ("ICH",), "include", ""),
    ("|hemorrhagic stroke|hypertensive|thalamus",             ("ICH",), "include", ""),
    ("|hemorrhagic stroke|hypertensive|cerebellar",           ("ICH",), "include", ""),
    ("|hemorrhagic stroke|hypertensive|internal capsule",     ("ICH",), "include", ""),
    ("|hemorrhagic stroke|hypertensive|pons",                 ("ICH",), "include", ""),
    ("|hemorrhagic stroke|left sided",                        ("ICH",), "include", ""),
    ("|hemorrhagic stroke|right sided",                       ("ICH",), "include", ""),
    ("|hemorrhagic stroke|cortical",                          ("ICH",), "include", ""),
    ("|hemorrhagic stroke|brainstem",                         ("ICH",), "include", ""),
    ("|hemorrhagic stroke|from amyloid angiopathy",           ("ICH",), "include", ""),
    # haemorrhagic transformation carries both, as it would in MIMIC (I63 + I61)
    ("|hemorrhagic stroke|transformation of ischemic stroke", ("AIS", "ICH"), "include",
     "carries both subtypes; Paper 1 precedence then labels it AIS"),
    ("|hemorrhagic stroke|into pre-existent CNS mass",        ("ICH",), "review",  "haemorrhage into a tumour, not primary stroke"),
    ("|hemorrhagic stroke|postoperative",                     ("ICH",), "review",  "post-surgical haemorrhage"),
    # ---- subarachnoid ----
    ("|hemorrhagic stroke|subarachnoid hemorrhage",                             ("SAH",), "include", ""),
    ("|hemorrhagic stroke|subarachnoid hemorrhage|from ruptured berry aneurysm",("SAH",), "include", ""),
    ("|hemorrhagic stroke|subarachnoid hemorrhage|from ruptured AV malformation",("SAH",), "include", ""),
    ("|hemorrhagic stroke|subarachnoid hemorrhage|from cavernous angioma",      ("SAH",), "include", ""),
    ("|hemorrhagic stroke|subarachnoid hemorrhage|with hydrocephalus",          ("SAH",), "include", "complication of SAH"),
    ("|hemorrhagic stroke|subarachnoid hemorrhage|with vasospasm",              ("SAH",), "include", "complication of SAH"),
    # ---- the trap ----
    ("|hemorrhagic stroke|subarachnoid hemorrhage|traumatic", (), "exclude",
     "TRAUMATIC SAH inside the stroke branch; a prefix match would admit it"),
]
TRAUMA_PREFIX = "neurologic|trauma - cns"

# APACHE admission diagnoses that exclude a stay, enumerated rather than matched
# on the word "trauma". A regex on "trauma" is wrong twice over: it catches
# cross-reference notes ("Hemorrhage ... (for trauma see Trauma)",
# "Fracture-pathological, non-union, non-traumatic, for fractures due to trauma
# see Trauma", "Facial surgery (if related to trauma, see Trauma)"), and it
# catches limb, chest, abdominal and pelvic injuries that Paper 1 never
# excluded -- a stroke patient with a fractured wrist is still a stroke patient.
#
# The list below is the eICU image of Paper 1's ICD exclusion prefixes:
#   800-804 skull and facial fracture, 850-854 intracranial injury, S06.
# Head and face injuries are in because 802 (facial bones) is in Paper 1's list;
# isolated spinal, chest, abdominal, pelvic and extremity trauma are out.
APACHE_TRAUMA_EXCLUDE = {
    "Head only trauma", "Head/abdomen trauma", "Head/chest trauma",
    "Head/extremity trauma", "Head/face trauma", "Head/multiple trauma",
    "Head/pelvis trauma", "Head/spinal trauma",
    "Hematoma, subdural", "Hematoma subdural, surgery for",
    "Hematoma, epidural", "Hematoma-epidural, surgery for",
    "Face only trauma", "Face only trauma, surgery for",
    "Face/multiple trauma", "Face/multiple trauma, surgery for",
    "Abdomen/face trauma", "Chest/face trauma", "Extremity/face trauma",
    "Extremity/face trauma, surgery for", "Pelvis/face trauma", "Spinal/face trauma",
}

# ---------------------------------------------------------------------------
dx = pd.read_csv(f"{B}/diagnosis.csv.gz",
                 usecols=["patientunitstayid", "diagnosisstring", "icd9code", "diagnosispriority"])
pat = pd.read_csv(f"{B}/patient.csv.gz",
                  usecols=["patientunitstayid", "hospitalid", "apacheadmissiondx"])
ds = dx.diagnosisstring.fillna("")
dsl = ds.str.lower()

rows, seen = [], {}
for suffix, subs, status, note in PATHS:
    full = P + suffix
    m = dsl == full.lower()
    stays = set(dx.loc[m, "patientunitstayid"])
    codes = sorted({c.strip() for s in dx.loc[m, "icd9code"].dropna() for c in str(s).split(",")})
    rows.append({"diagnosisstring": full, "subtypes": "+".join(subs) or "-", "status": status,
                 "n_stays": len(stays), "icd_codes_attached": ", ".join(codes), "note": note,
                 "freeze_date": FREEZE_DATE})
    if status != "exclude":
        for s in stays:
            seen.setdefault(s, {"subs": set(), "review": False})
            seen[s]["subs"].update(subs)
            if status == "review":
                seen[s]["review"] = True
frozen = pd.DataFrame(rows)
frozen.to_csv(OUT / "phenotype_paths_frozen.csv", index=False)

# every stroke-branch path must appear in the frozen table -- guard against a
# future eICU release adding one that this file has never been reviewed against
observed = set(ds[dsl.str.startswith(P.lower())].unique())
missing = observed - set(frozen.diagnosisstring)
if missing:
    raise SystemExit(f"UNREVIEWED PATHS in the stroke branch:\n  " + "\n  ".join(sorted(missing)))
print(f"Frozen path table: {len(frozen)} paths, all {len(observed)} observed stroke paths accounted for")
print(f"  include {(frozen.status=='include').sum()} | review {(frozen.status=='review').sum()} "
      f"| exclude {(frozen.status=='exclude').sum()}")

# ---------------------------------------------------------------------------
# exclusions
# ---------------------------------------------------------------------------
trauma_dx = set(dx.loc[dsl.str.startswith(TRAUMA_PREFIX), "patientunitstayid"])
trauma_apache = set(pat.loc[pat.apacheadmissiondx.isin(APACHE_TRAUMA_EXCLUDE), "patientunitstayid"])
traumatic_sah = set(dx.loc[dsl == (P + "|hemorrhagic stroke|subarachnoid hemorrhage|traumatic").lower(),
                           "patientunitstayid"])
excluded = trauma_dx | trauma_apache | traumatic_sah
print(f"\nExclusion pool: {len(trauma_dx):,} CNS-trauma path, {len(trauma_apache):,} APACHE trauma dx, "
      f"{len(traumatic_sah):,} traumatic SAH inside the stroke branch")

# spinal-cord-injury-only stays are a deliberate, quantified broadening vs Paper 1,
# whose ICD exclusion covered skull fracture and intracranial injury but not 806/952
sci_only = {s for s in trauma_dx
            if not any(("intracranial injury" in p.lower()) or ("fracture of skull" in p.lower())
                       for p in dx.loc[dx.patientunitstayid == s, "diagnosisstring"].fillna(""))}
print(f"  of the CNS-trauma stays, {len(sci_only & set(seen)):,} stroke stays are excluded on a "
      f"non-intracranial (mostly spinal) trauma path -- a broadening vs Paper 1, reported as a deviation")

# ---------------------------------------------------------------------------
# stay-level phenotype, Paper 1 subtype precedence
# ---------------------------------------------------------------------------
def label(subs):
    if "ICH" in subs and "SAH" in subs:
        return "ICH+SAH"
    for s in ("AIS", "ICH", "SAH"):          # Paper 1 append order
        if s in subs:
            return s
    return "unspecified"

recs = [{"patientunitstayid": sid,
         "stroke_subtype": label(v["subs"]),
         "unspecified_only": int(v["subs"] == {"unspecified"}),
         "has_review_path": int(v["review"]),
         "excluded_trauma": int(sid in excluded)}
        for sid, v in seen.items()]
stays = pd.DataFrame(recs).sort_values("patientunitstayid")
stays = stays.merge(pat[["patientunitstayid", "hospitalid", "apacheadmissiondx"]],
                    on="patientunitstayid", how="left")
stays.to_csv(OUT / "eicu_stroke_stays.csv", index=False)

keep = stays[stays.excluded_trauma == 0]
print(f"\nStroke stays identified: {len(stays):,}; after trauma exclusion: {len(keep):,}")
print("\nSubtype distribution (after trauma exclusion):")
print(keep.stroke_subtype.value_counts().to_string())
print(f"\nunspecified-only stays (dropped by the pre-specified sensitivity): {keep.unspecified_only.sum():,}")
print(f"stays touching a 'review' path (perioperative / postoperative / into CNS mass): {keep.has_review_path.sum():,}")

# ---------------------------------------------------------------------------
# independent cross-check
# ---------------------------------------------------------------------------
ad = pat.apacheadmissiondx.fillna("")
apache_stroke = set(pat.loc[ad.str.contains(
    r"CVA, cerebrovascular|Hemorrhage/hematoma, intracranial|Subarachnoid hemorrhage",
    case=False, regex=True), "patientunitstayid"])
sset = set(keep.patientunitstayid)
cc = pd.DataFrame([
    {"agreement": "both string path and APACHE admission dx", "n": len(sset & apache_stroke)},
    {"agreement": "string path only", "n": len(sset - apache_stroke)},
    {"agreement": "APACHE admission dx only", "n": len(apache_stroke - sset)},
])
cc.to_csv(OUT / "phenotype_crosscheck.csv", index=False)
print("\nCross-check against the one independent source (apacheadmissiondx):")
print(cc.to_string(index=False))
print(f"  sensitivity of the string phenotype vs APACHE: {len(sset & apache_stroke)/len(apache_stroke)*100:.1f}%")
print(f"\nWrote {OUT}/")
