"""
Paper 2: write each manuscript table to its own .docx, extracted from the
manuscript itself.

Journals want one file per table. Script 25 deliberately wrote the tables into
the prose sources rather than rendering them standalone, so that this step can
copy the <w:tbl> XML out of the built document. That removes drift by
construction: a standalone file cannot say anything the manuscript does not,
because it is the same XML. Rendering from the analysis CSVs is what shipped a
divergent artefact in Paper 1 and is not repeated here.

Usage:  python 28_extract_tables_from_manuscript.py
"""
import re
import unicodedata
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
# The nine-table draft this was written for. JAMIA takes four tables and five
# supplementary ones, built by 41 and 42, so this script and its output are the
# record of the earlier structure rather than part of the submission pipeline.
SRC = HERE / "manuscript/_superseded/manuscript_draft_v7_EN.docx"
OUT = HERE / "submission_superseded_9tables/tables"

MD = HERE / "manuscript/tables_all.md"

CAPTION = re.compile(r"^Table\s+(\d+[ab]?)\.")
BLOCK = re.compile(r"<w:(p|tbl)[ >].*?</w:\1>", re.S)
# <w:t> only -- <w:t[^>]*> also matches <w:tbl>, <w:tr>, <w:tc> and <w:tcPr>.
WT = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)


def text_of(blob):
    return "".join(WT.findall(blob)).strip()


def norm(s):
    """Compare content, not typography. pandoc curls quotes, and markdown emphasis
    markers become real italics in the document, so `*t*` there is `t` here."""
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*\s][^*]*)\*", r"\1", s)
    s = unicodedata.normalize("NFKC", s)
    for a, b in [("\u2013", "-"), ("\u2014", "-"), ("\u2212", "-"),
                 ("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


OUT.mkdir(parents=True, exist_ok=True)
doc = zipfile.ZipFile(SRC).read("word/document.xml").decode("utf-8")
head = doc[: doc.index("<w:body>")]           # <w:document ...> with all namespaces
body = re.search(r"<w:body>(.*)</w:body>", doc, re.S).group(1)

# Keep the page size and margins; drop the running head and page numbers, which
# belong to the manuscript rather than to a table file.
sect = re.search(r"<w:sectPr[ >].*?</w:sectPr>", body, re.S)
sect = re.sub(r"<w:(header|footer)Reference[^>]*/>", "", sect.group(0)) if sect else ""

# Walk the body in order: a caption paragraph claims the next table after it.
blocks = [(m.group(1), m.group(0)) for m in BLOCK.finditer(body)]
tables, pending = [], None
for tag, blob in blocks:
    if tag == "p":
        m = CAPTION.match(text_of(blob))
        if m:
            pending = (m.group(1), blob)
    elif pending:
        tables.append((*pending, blob))
        pending = None

if not tables:
    raise SystemExit("no captioned tables found -- has the manuscript layout changed?")

groups = {}
for num, cap, tbl in tables:
    groups.setdefault(num.rstrip("ab") if num[-1] in "ab" else num, []).append((cap, tbl))

spacer = '<w:p><w:pPr><w:spacing w:before="0" w:after="240"/></w:pPr></w:p>'

# Read the source package once. Re-reading members from a single live handle
# across several output files is what raised BadZipFile here.
with zipfile.ZipFile(SRC) as src_zip:
    members = [(item, src_zip.read(item.filename)) for item in src_zip.infolist()
               if not item.filename.startswith("word/media/")]
for num in sorted(groups, key=lambda s: int(s)):
    panels = groups[num]
    parts = []
    for i, (cap, tbl) in enumerate(panels):
        if i:
            parts.append(spacer)
        parts.extend([cap, tbl])
    xml = f"{head}<w:body>{''.join(parts)}{sect}</w:body></w:document>"

    dest = OUT / f"Table_{num}.docx"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for item, data in members:           # figures already excluded
            z.writestr(item, xml.encode("utf-8")
                       if item.filename == "word/document.xml" else data)
    rows = len(re.findall(r"<w:tr[ >]", panels[0][1]))
    kb = dest.stat().st_size / 1024
    print(f"  {dest.name:<14} {rows:2d} rows  {kb:6.1f} KB  {text_of(panels[0][0])[:62]}")

print(f"\nwrote {len(groups)} table files to {OUT}")

# --- verify: every cell must match the markdown the manuscript was built from --
def docx_cells(path):
    x = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    t = re.findall(r"<w:tbl>.*?</w:tbl>", x, re.S)
    assert len(t) == 1, f"{path.name}: {len(t)} tables"
    return [[text_of(tc) for tc in re.findall(r"<w:tc[ >].*?</w:tc>", tr, re.S)]
            for tr in re.findall(r"<w:tr[ >].*?</w:tr>", t[0], re.S)]


src = {}
for chunk in re.split(r"\n(?=\*\*Table )", MD.read_text()):
    m = re.match(r"\*\*Table (\d+)\.\*\*", chunk)
    if m:
        src[m.group(1)] = [[c.strip() for c in ln.strip().strip("|").split("|")]
                           for ln in chunk.splitlines()
                           if ln.strip().startswith("|")
                           and not re.match(r"^\|[\s\-|]+\|$", ln.strip())]

problems, checked = [], 0
for num in sorted(groups, key=lambda s: int(s)):
    got, want = docx_cells(OUT / f"Table_{num}.docx"), src.get(num)
    if want is None:
        problems.append(f"Table {num}: not in {MD.name}")
        continue
    if len(got) != len(want):
        problems.append(f"Table {num}: {len(got)} rows, markdown has {len(want)}")
        continue
    for i, (rg, rw) in enumerate(zip(got, want)):
        for j, (a, b) in enumerate(zip(rg, rw)):
            checked += 1
            if norm(a) != norm(b):
                problems.append(f"Table {num} r{i}c{j}: docx={a!r} md={b!r}")

# --- verify: tables are numbered in order of first citation -------------------
seen = []
for tag, blob in blocks:
    if tag != "p":
        continue
    t = text_of(blob)
    if CAPTION.match(t):
        continue
    for n in re.findall(r"\bTable\s+(\d+)\b", t):
        if n not in seen:
            seen.append(n)
if seen != sorted(seen, key=int):
    problems.append("citation order is " + " ".join(seen) + ", not ascending")

print(f"verified {checked} cells against {MD.name}; "
      f"first-citation order {' '.join(seen)}")
if problems:
    raise SystemExit("FAILED\n  " + "\n  ".join(problems))
print("all table files match the manuscript and are numbered in citation order")
