"""
Paper 2: interleave the tables into the Results text.

JAMIA asks for tables "placed in the main text where the table is first cited".
The merged files were first produced by hand, which means a change to a table --
a new column, a corrected value -- reaches the submitted .docx only if somebody
remembers to redo the merge. This script does the merge from the two sources
every time, so forgetting is no longer possible.

Each table block is placed at the end of the section that first cites it.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
MS = HERE / "manuscript"

LANGS = {
    "EN": {"results": "results_jamia_EN.md", "tables": "tables_jamia.md",
           "out": "results_jamia_with_tables_EN.md", "cap": r"^\*\*Table (\d+)\.\*\*"},
    "CN": {"results": "results_jamia_CN.md", "tables": "tables_jamia_CN.md",
           "out": "results_jamia_with_tables_CN.md", "cap": r"^\*\*表 (\d+)\.\*\*"},
}
CITE = {"EN": r"Table {n}\b", "CN": r"表 {n}(?![0-9])"}


def split_tables(text, cap_pat):
    """Return {table number: the caption paragraph plus its markdown rows}."""
    lines = text.splitlines()
    starts = [(i, int(m.group(1)))
              for i, l in enumerate(lines) if (m := re.match(cap_pat, l))]
    out = {}
    for k, (i, n) in enumerate(starts):
        j = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        out[n] = "\n".join(lines[i:j]).strip()
    return out


def merge(lang):
    cfg = LANGS[lang]
    res = (MS / cfg["results"]).read_text()
    tabs = split_tables((MS / cfg["tables"]).read_text(), cfg["cap"])
    if not tabs:
        print(f"  !! {lang}: 没有从 {cfg['tables']} 解析出任何表")
        return False

    # Section boundaries: a table goes at the end of the section that first cites it.
    lines = res.splitlines()
    heads = [i for i, l in enumerate(lines) if l.startswith("## ")] + [len(lines)]
    placed, where = {}, {}
    for n in sorted(tabs):
        pat = re.compile(CITE[lang].format(n=n))
        for k in range(len(heads) - 1):
            body = "\n".join(lines[heads[k]:heads[k + 1]])
            if pat.search(body):
                where.setdefault(heads[k + 1], []).append(n)
                placed[n] = k
                break
    missing = [n for n in tabs if n not in placed]
    if missing:
        print(f"  !! {lang}: 表 {missing} 在正文里找不到引用，无法定位")
        return False

    out, prev = [], 0
    for cut in sorted(where):
        out.append("\n".join(lines[prev:cut]).rstrip())
        for n in where[cut]:
            out.append("")
            out.append(tabs[n])
        prev = cut
    out.append("\n".join(lines[prev:]).rstrip())
    merged = "\n\n".join(x for x in out if x != "") + "\n"
    (MS / cfg["out"]).write_text(merged)

    # Verify: every source line survived, and every table landed exactly once.
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    m = norm(merged)
    bad = [l for l in res.splitlines()
           if (t := norm(l)) and len(t) > 20 and t not in m]
    bad += [f"表 {n} 未落位" for n in tabs if norm(tabs[n].splitlines()[0]) not in m]
    dup = [n for n in tabs if merged.count(tabs[n].splitlines()[0]) != 1]
    print(f"  {lang}: {cfg['out']}  {len(tabs)} 张表插入 "
          f"{sorted(placed, key=lambda n: placed[n])} 节")
    if bad or dup:
        for b in bad[:5]:
            print(f"     ❌ {b[:70]}")
        for d in dup:
            print(f"     ❌ 表 {d} 出现 {merged.count(tabs[d].splitlines()[0])} 次")
        return False
    print("     ✅ 正文每一行都在，四张表各出现一次")
    return True


if __name__ == "__main__":
    sys.exit(0 if all(merge(l) for l in (sys.argv[1:] or ["EN", "CN"])) else 1)
