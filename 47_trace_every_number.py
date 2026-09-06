"""
Paper 2: trace every number printed in the JAMIA submission back to a pipeline output.

Script 23 runs source -> text: it takes a number it already knows and asks whether
the manuscript still says it. That direction cannot see a number the manuscript
prints which no pipeline ever produced -- a stale value left over from an earlier
run, a typo in a table cell, a figure legend nobody re-derived. Those are exactly
the numbers a reviewer recomputes.

This script runs text -> source. Every numeric token in every file that goes into
the submission is looked up in a pool built from every CSV the pipeline wrote.
A token with no match is not necessarily wrong, but it is untraceable, and
untraceable is the state a number should never be submitted in.
"""
import re
import sys
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
MS = HERE / "manuscript"

SUBMISSION = [
    "title_page_EN.md", "abstract_jamia_EN.md", "intro_jamia_EN.md",
    "methods_jamia_EN.md", "results_jamia_EN.md", "discussion_jamia_EN.md",
    "tables_jamia.md", "supplementary_tables.md", "figure_legends_jamia_EN.md",
    "data_availability_EN.md", "declarations_EN.md",
]

# A number in the prose is a claim; a number inside these is not. Reference years
# and page ranges, software versions, and the digits inside an identifier are
# typography, not results.
STRIP = [
    (re.compile(r"^##+ .*$", re.M), ""),            # heading numbers (Table 1, Figure 2)
    (re.compile(r"`[^`]*`"), " "),                  # inline code / column names
    (re.compile(r"\[\d+\]"), " "),                  # citation markers
    (re.compile(r"https?://\S+"), " "),             # URLs and DOIs
]

NUM = re.compile(r"(?<![A-Za-z0-9_.])[−-]?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9_])")


def tokens(text):
    for pat, rep in STRIP:
        text = pat.sub(rep, text)
    out = []
    for m in NUM.finditer(text):
        raw = m.group(0)
        val = float(raw.replace(",", "").replace("−", "-"))
        lo = max(0, m.start() - 70)
        out.append((raw, val, " ".join(text[lo:m.end() + 40].split())))
    return out


OUT = HERE / "audit/number_provenance.csv"


def build_pool():
    """Every numeric value the pipeline wrote, at every precision it could print at."""
    pool = {}
    for csv in sorted(HERE.rglob("*.csv")):
        # This script's own output holds every manuscript number in a `value`
        # column. Left in the pool it would certify each number against itself
        # on the second run, and the check would report 100% for ever after.
        if "paper2_code_repository" in csv.parts or csv == OUT:
            continue
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        rel = str(csv.relative_to(HERE))
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            # Cap per column: assignment files hold hundreds of thousands of rows
            # and would match anything by sheer coverage.
            if len(s) > 5000:
                s = pd.Series(sorted(set(s.round(6))))
                if len(s) > 5000:
                    continue
            for v in s.unique():
                for nd in range(0, 5):
                    pool.setdefault(round(float(v), nd), set()).add(f"{rel}:{col}")
                # percentages written as 63.3 from a proportion stored as 0.633
                for nd in range(0, 3):
                    pool.setdefault(round(float(v) * 100, nd), set()).add(f"{rel}:{col}")
    return pool


def main():
    pool = build_pool()
    print(f"来源池：{len(pool)} 个不同的数值，取自 {len(list(HERE.rglob('*.csv')))} 个 CSV\n")

    rows = []
    for f in SUBMISSION:
        p = MS / f
        if not p.exists():
            print(f"!! {f} 不存在")
            continue
        for raw, val, ctx in tokens(p.read_text()):
            src = pool.get(round(val, 6)) or pool.get(round(val, 4)) or pool.get(round(val, 3))
            rows.append({"file": f, "raw": raw, "value": val,
                         "traced": bool(src),
                         "source": sorted(src)[0] if src else "",
                         "n_sources": len(src) if src else 0,
                         "context": ctx})

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    out = OUT

    tot, ok = len(R), int(R.traced.sum())
    print(f"提交稿中的数字：{tot}")
    print(f"  可追溯到流程输出：{ok}  ({ok/tot*100:.1f}%)")
    print(f"  追不到：          {tot-ok}\n")
    print("按文件：")
    g = R.groupby("file").traced.agg(["size", "sum"])
    for f in SUBMISSION:
        if f in g.index:
            n, k = int(g.loc[f, "size"]), int(g.loc[f, "sum"])
            print(f"  {f:34s} {k:4d}/{n:<4d} 追不到 {n-k}")
    print(f"\n逐条写入 {out.relative_to(HERE)}")
    return R


if __name__ == "__main__":
    main()
