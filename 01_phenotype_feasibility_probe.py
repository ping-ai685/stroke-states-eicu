"""
Paper 2, step 1: feasibility probe of the eICU stroke phenotype.

Run BEFORE the phenotype is frozen. It answers three questions that decide how
section 5.5 of the protocol has to be written:

  1. How many stroke ICU stays does each candidate phenotype yield?
  2. Are diagnosis.icd9code and diagnosis.diagnosisstring independent evidence?
  3. After the Paper 1 filters, how many patients and hospitals remain, and is a
     hospital-level analysis powered?

Findings on 23 Aug 2026 are recorded in PROTOCOL_REVIEW_v1.0.md. The headline is
that eICU's icd9code is a deterministic re-encoding of diagnosisstring, so the
two cannot cross-validate each other, and Paper 1's ICD code list must NOT be
transported literally.

Reads eICU only; writes nothing.
"""
import pandas as pd
from collections import Counter

B = "/Volumes/Lexar/research data/eICU dataset/eicu-collaborative-research-database-2.0"

dx = pd.read_csv(f"{B}/diagnosis.csv.gz",
                 usecols=["patientunitstayid", "diagnosisstring", "icd9code", "diagnosispriority"])
pat = pd.read_csv(f"{B}/patient.csv.gz",
                  usecols=["patientunitstayid", "patienthealthsystemstayid", "uniquepid", "age",
                           "hospitalid", "unittype", "unitvisitnumber", "unitdischargeoffset",
                           "unitdischargestatus", "hospitaldischargestatus", "apacheadmissiondx"])
ds = dx.diagnosisstring.fillna("").str.lower()
print(f"eICU v2.0: {len(pat):,} unit stays, {pat.uniquepid.nunique():,} patients, "
      f"{pat.hospitalid.nunique()} hospitals, {len(dx):,} diagnosis rows\n")

# ---------------------------------------------------------------------------
# 1. three candidate phenotypes
# ---------------------------------------------------------------------------
ICD9_AIS = {"43301","43311","43321","43331","43381","43391","43401","43411","43491"}
def codes(s):
    return [c.strip().replace(".", "").upper() for c in str(s).split(",")] if pd.notna(s) else []
def is_stroke(cl):
    return any(c in ICD9_AIS or c.startswith(("I63",)) or c in {"431", "430"}
               or c.startswith(("I61", "I60")) for c in cl)

dx["_codes"] = dx.icd9code.map(codes)
icd_ids = set(dx[dx._codes.map(is_stroke)].patientunitstayid)
str_ids = set(dx[ds.str.startswith("neurologic|disorders of vasculature|stroke")].patientunitstayid)
adm_ids = set(pat[pat.apacheadmissiondx.fillna("").str.contains(
    r"CVA, cerebrovascular|Hemorrhage/hematoma, intracranial|Subarachnoid hemorrhage",
    case=False, regex=True)].patientunitstayid)
print("Candidate phenotypes (unit stays):")
print(f"  Paper 1 ICD code set, applied literally : {len(icd_ids):,}")
print(f"  diagnosisstring stroke path             : {len(str_ids):,}")
print(f"  apacheadmissiondx                       : {len(adm_ids):,}")

# ---------------------------------------------------------------------------
# 2. is icd9code independent of diagnosisstring?
# ---------------------------------------------------------------------------
print(f"\nicd9code populated on {dx.icd9code.notna().mean()*100:.1f}% of diagnosis rows; "
      f"{(dx.groupby('patientunitstayid').icd9code.apply(lambda s: s.notna().any())).mean()*100:.0f}% of stays have >=1 coded row")
print("Codes attached to each stroke diagnosisstring path -- if each path maps to one")
print("fixed code pair, the two fields are the same evidence, not two sources:")
for label, mask in [
    ("ischemic",     ds.str.startswith("neurologic|disorders of vasculature|stroke|ischemic")),
    ("hemorrhagic",  ds.str.startswith("neurologic|disorders of vasculature|stroke|hemorrhagic")),
    ("unspecified",  ds == "neurologic|disorders of vasculature|stroke"),
]:
    c = Counter(code.strip() for s in dx.loc[mask, "icd9code"].dropna() for code in str(s).split(","))
    top = ", ".join(f"{k} ({v:,})" for k, v in c.most_common(4))
    print(f"  {label:<12} {mask.sum():>7,} rows -> {top}")

# ---------------------------------------------------------------------------
# 3. cohort flow under the Paper 1 filters, string-based phenotype
# ---------------------------------------------------------------------------
AIS = ds.str.startswith("neurologic|disorders of vasculature|stroke|ischemic")
HEM = ds.str.startswith("neurologic|disorders of vasculature|stroke|hemorrhagic")
UNS = ds == "neurologic|disorders of vasculature|stroke"
SAH = ds.str.contains(r"hemorrhagic stroke\|subarachnoid hemorrhage", regex=True)
TRAUMA = ds.str.startswith("neurologic|trauma - cns")
sub = {"AIS": set(dx[AIS].patientunitstayid), "SAH": set(dx[SAH].patientunitstayid),
       "ICH": set(dx[HEM & ~SAH].patientunitstayid), "unspecified": set(dx[UNS].patientunitstayid)}
print("\nSubtype counts (a stay may carry more than one):")
for k, v in sub.items():
    print(f"  {k:<12} {len(v):,}")

ids = set().union(*sub.values()) - set(dx[TRAUMA].patientunitstayid)
p = pat[pat.patientunitstayid.isin(ids)].copy()
p["age_n"] = pd.to_numeric(p.age.replace("> 89", "90"), errors="coerce")
steps = [("1. stroke path, CNS-trauma paths excluded", p)]
p = p[p.age_n >= 18];                        steps.append(("2. age >=18", p))
p = p[p.unitvisitnumber == 1];               steps.append(("3. first ICU unit stay", p))
p = p[p.unitdischargeoffset >= 720];         steps.append(("4. ICU LOS >=12 h", p))
p = p.sort_values(["uniquepid", "patienthealthsystemstayid"]).drop_duplicates("uniquepid")
steps.append(("5. one hospitalization per patient", p))
print("\nCohort flow:")
for label, d in steps:
    print(f"  {label:<42} {len(d):>6,} stays  {d.hospitalid.nunique():>4} hospitals")

print(f"\nICU mortality {(p.unitdischargestatus=='Expired').mean()*100:.1f}%  "
      f"hospital mortality {(p.hospitaldischargestatus=='Expired').mean()*100:.1f}%  "
      f"median age {p.age_n.median():.0f}  median ICU LOS {p.unitdischargeoffset.median()/1440:.2f} d")
print(f"Excluding the unspecified-stroke path instead gives {len(p) - len(p[p.patientunitstayid.isin(sub['unspecified'])]):,} patients")

h = p.groupby("hospitalid").size()
print(f"\nHospital-level feasibility: {h.size} hospitals, median {h.median():.0f} patients "
      f"(IQR {h.quantile(.25):.0f}-{h.quantile(.75):.0f}, max {h.max()})")
for t in (20, 50, 100, 150):
    print(f"  >={t:>3} patients: {(h>=t).sum():>3} hospitals covering {h[h>=t].sum():>5,} "
          f"({h[h>=t].sum()/h.sum()*100:.0f}% of the cohort)")
