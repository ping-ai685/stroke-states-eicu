"""
Paper 2, checklist item 5b: validate the returned term lists before freezing.

The review is not accepted on trust any more than the automatic proposal was.
Three things are checked:

  1. completeness and legality -- no blank decisions, no values outside the
     allowed set for each list;
  2. what the reviewer changed relative to the proposal, listed in full, because
     those rows are where the automatic rule was wrong and they are what the
     Methods section will have to describe;
  3. INTERNAL CONSISTENCY against the rule the reviewer themselves set --
     sedation and vasoactive support require drug identity AND continuous-infusion
     evidence. Any row marked `sedative` or `vasoactive` on a drug whose orders
     are not continuous is surfaced, not silently accepted.

Reads review_returned/, writes review_returned/validation_report.csv.
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "review_returned"
LEGAL = {
    "drugs": {"vasoactive", "sedative", "exclude", "exclude (vasodilator)"},
    "urine": {"include", "exclude"},
    "vent": {"INVASIVE vent", "NON-invasive (exclude)", "CRRT", "dialysis-not-CRRT (exclude)",
             "vent-setting evidence", "not evidence"},
}
FILES = {"drugs": ("termlist_1_drugs_REVIEWED.csv", "proposed_class", "drugname"),
         "urine": ("termlist_2_urine_sources_REVIEWED.csv", "proposed_include", "celllabel"),
         "vent": ("termlist_3_ventilation_crrt_REVIEWED.csv", "proposed", "treatmentstring")}

issues, tables = [], {}
for key, (fname, propcol, namecol) in FILES.items():
    d = pd.read_csv(R / fname, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    d = d[d[namecol].notna()]
    tables[key] = d
    dec = d.REVIEWER_DECISION.fillna("").str.strip()
    print(f"\n{'='*70}\n{key.upper()}  ({len(d)} rows)")
    n_blank = int((dec == "").sum())
    illegal = sorted(set(dec) - LEGAL[key] - {""})
    print(f"  blank decisions : {n_blank}")
    print(f"  illegal values  : {illegal if illegal else 'none'}")
    if n_blank: issues.append({"list": key, "issue": "blank decisions", "n": n_blank, "detail": ""})
    if illegal: issues.append({"list": key, "issue": "illegal decision values", "n": len(illegal),
                               "detail": "; ".join(illegal)})
    print("  decision counts:")
    for k, v in dec.value_counts().items():
        print(f"    {k:<32}{v:>5}")
    prop = d[propcol].fillna("").str.strip()
    changed = d[dec.values != prop.values]
    print(f"  rows where the reviewer overrode the proposal: {len(changed)}")
    if len(changed):
        show = changed[[namecol, propcol, "REVIEWER_DECISION"]].copy()
        show["stays"] = changed.get("stays")
        print(show.sort_values("stays", ascending=False).head(30).to_string(index=False))

# --- internal consistency: the reviewer's own continuous-infusion rule --------
d = tables["drugs"]
dec = d.REVIEWER_DECISION.fillna("").str.strip()
flagged = d[dec.isin(["vasoactive", "sedative"])].copy()
flagged["cont"] = pd.to_numeric(flagged.pct_orders_continuous, errors="coerce")
bad = flagged[(flagged.cont.notna()) & (flagged.cont < 50)]
print(f"\n{'='*70}\nINTERNAL CONSISTENCY -- reviewer's rule: identity AND continuous infusion")
print(f"  rows marked vasoactive/sedative whose orders are <50% continuous: {len(bad)}")
if len(bad):
    print(bad[["drugname", "source_table", "frequency", "cont", "stays",
               "REVIEWER_DECISION"]].sort_values("stays", ascending=False).to_string(index=False))
    issues.append({"list": "drugs", "issue": "marked active but orders mostly bolus",
                   "n": len(bad), "detail": "; ".join(bad.drugname.head(8))})
kept = flagged[~flagged.index.isin(bad.index)]
print(f"  rows retained as active support: {len(kept)} "
      f"({(kept.REVIEWER_DECISION=='vasoactive').sum()} vasoactive, "
      f"{(kept.REVIEWER_DECISION=='sedative').sum()} sedative)")

pd.DataFrame(issues if issues else [{"list": "-", "issue": "none", "n": 0, "detail": ""}]) \
  .to_csv(R / "validation_report.csv", index=False)
print(f"\n{'='*70}\nissues found: {len(issues)}")
