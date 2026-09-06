"""
Paper 2: assemble the JAMIA submission .docx from its section files.

The first build was a pandoc command typed at a shell. That is fine once and a
liability afterwards: nothing records which section files went in, in which
order, so a rebuild months later can silently differ from what was submitted.
This script fixes the order in code, rebuilds both languages, and then verifies
that what came out contains what went in.
"""
import re
import subprocess
import sys
import zipfile
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pandoc_path import pandoc   # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
merge_results = import_module("51_merge_results_with_tables").merge
MS = HERE / "manuscript"

ORDER = {
    "EN": ["title_page_EN.md", "abstract_jamia_EN.md", "intro_jamia_EN.md",
           "methods_jamia_EN.md", "results_jamia_with_tables_EN.md",
           "discussion_jamia_EN.md", "data_availability_EN.md",
           "declarations_EN.md", "references_jamia_EN.md",
           "figure_legends_jamia_EN.md"],
    "CN": ["title_page_CN.md", "abstract_jamia_CN.md", "intro_jamia_CN.md",
           "methods_jamia_CN.md", "results_jamia_with_tables_CN.md",
           "discussion_jamia_CN.md", "data_availability_CN.md",
           "declarations_CN.md", "references_jamia_CN.md",
           "figure_legends_jamia_CN.md"],
}


def fit_tables(path):
    """Let Word size table columns by content, across the full text width.

    pandoc writes a fixed layout with every column the same width, so a nine-column
    table gives 0.6 inch to a cell holding "vent-ascertainable" and the same 0.6
    inch to one holding "9.3". The result reads as if the table has burst its
    margins. Autofit plus a full-width table lets the wide columns take the room
    the narrow ones do not need.
    """
    import shutil, tempfile
    z = zipfile.ZipFile(path)
    doc = z.read("word/document.xml").decode()
    members = {n: z.read(n) for n in z.namelist()}
    z.close()

    def one(m):
        t = m.group(0)
        t = re.sub(r"<w:tblW[^>]*/>", "", t)
        t = re.sub(r"<w:tblLayout[^>]*/>", "", t)
        ins = '<w:tblW w:w="5000" w:type="pct"/><w:tblLayout w:type="autofit"/>'
        if "<w:tblPr>" in t:
            t = t.replace("<w:tblPr>", "<w:tblPr>" + ins, 1)
        elif "<w:tblPr/>" in t:
            t = t.replace("<w:tblPr/>", "<w:tblPr>" + ins + "</w:tblPr>", 1)
        return t

    doc, n = re.subn(r"<w:tbl>.*?</w:tbl>", one, doc, flags=re.S)
    members["word/document.xml"] = doc.encode()
    tmp = tempfile.mktemp(suffix=".docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, data in members.items():
            zo.writestr(name, data)
    shutil.move(tmp, path)
    return n


def docx_text(path):
    """The document's text, one line per paragraph.

    A paragraph is split into runs whenever formatting changes, so "PhD^1,2^"
    is three runs. Runs must be joined with nothing and paragraphs with a
    newline; joining every run with a space inserts gaps that no source line
    contains, and the comparison then fails on formatting alone.
    """
    x = zipfile.ZipFile(path).read("word/document.xml").decode()
    # <w:t> only. <w:t[^>]*> would also match <w:tbl>, <w:tr>, <w:tc>, <w:tcPr>.
    RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
    paras = re.findall(r"<w:p[ >].*?</w:p>", x, re.S)
    return "\n".join("".join(RUN.findall(p)) for p in paras)


def norm(s):
    """Compare meaning, not typography.

    pandoc renders ^1,2^ as a superscript and turns ' into a curly quote, so a
    source line and its rendered form differ character for character while
    saying the same thing. Everything stripped here is presentation.
    """
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*\s][^*]*)\*", r"\1", s)
    s = re.sub(r"\^([^^]*)\^", r"\1", s)          # superscript markers
    s = s.replace("`", "")                        # inline code
    # A **bold** span may straddle a soft line break, leaving one stray ** on
    # each half that the paired patterns above cannot see.
    s = s.replace("*", "")
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    for a, b in [("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'),
                 ("\u201d", '"'), ("\u2014", "-"), ("\u2013", "-"),
                 ("\u2212", "-"), ("\u00a0", " ")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def build(lang):
    # The results file carries the tables inline, and it is generated. Rebuild it
    # first: otherwise a corrected table reaches tables_jamia.md and never reaches
    # the .docx, and nothing would report the discrepancy.
    if not merge_results(lang):
        print(f"  !! {lang}: 表格合并失败，未生成 docx")
        return False
    files = [MS / f for f in ORDER[lang]]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"  !! {lang}: 缺少 {missing}")
        return False
    out = MS / f"manuscript_JAMIA_{lang}.docx"
    # JAMIA: "all submissions should be double-spaced". Bare pandoc output is
    # single-spaced, so the reference document is not cosmetic -- without it the
    # file does not meet the journal's stated requirement.
    ref = MS / "jamia_reference.docx"
    assert ref.exists(), f"缺少 {ref.name}，无法保证双倍行距"
    subprocess.run([pandoc(), *[str(f) for f in files],
                    f"--reference-doc={ref}", "-o", str(out)], check=True)

    # Verify: every non-table line of every source file survived into the docx.
    got = norm(docx_text(out))
    bad = []
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("|"):
                continue
            n = norm(line)
            if len(n) > 25 and n not in got:
                bad.append((f.name, n[:70]))
    n_tbl = fit_tables(out)

    # Verify the requirement, do not assume the reference document carried.
    st = zipfile.ZipFile(out).read("word/styles.xml").decode()
    print(f"  {lang}: {out.name}  {out.stat().st_size:,} bytes  "
          f"来自 {len(files)} 个文件")
    body = sorted(set(re.findall(r'<w:style [^>]*w:styleId="BodyText".*?'
                                r'<w:spacing[^>]*w:line="(\d+)"', st, re.S)))
    cell = sorted(set(re.findall(r'<w:style [^>]*w:styleId="Compact".*?'
                                r'<w:spacing[^>]*w:line="(\d+)"', st, re.S)))
    pg = re.search(r'<w:pgSz[^>]*/>', zipfile.ZipFile(out).read(
        "word/document.xml").decode())
    print(f"     {'✅' if body == ['480'] else '❌'} 正文双倍行距 BodyText={body}")
    print(f"     {'✅' if cell == ['240'] else '❌'} 表格单倍行距 Compact={cell}")
    print(f"     {'✅' if pg else '❌'} 页面尺寸 {pg.group(0) if pg else '缺失'}")
    print(f"     ✅ {n_tbl} 张表已设为满宽自适应")
    if body != ["480"] or cell != ["240"] or not pg:
        return False
    if bad:
        print(f"     ❌ {len(bad)} 段没有进入 docx:")
        for f, t in bad[:6]:
            print(f"        {f}: {t}")
        return False
    print("     ✅ 每一段都在 docx 里找得到")
    return True




SINGLES = ["supplementary_tables.md", "cover_letter_JAMIA.md"]


def build_single(name):
    """The supplementary tables and the cover letter are separate uploads.

    They were being produced by a bare pandoc call, so they carried neither the
    page setup nor the table fitting the manuscript gets. A reviewer opens all
    three; they should look like one submission.
    """
    src = MS / name
    if not src.exists():
        print(f"  !! 缺少 {name}")
        return False
    out = src.with_suffix(".docx")
    ref = MS / "jamia_reference.docx"
    subprocess.run([pandoc(), str(src), f"--reference-doc={ref}", "-o", str(out)],
                   check=True)
    n = fit_tables(out)
    print(f"  {out.name}  {out.stat().st_size:,} bytes  {n} 张表已设为满宽自适应")
    return True


if __name__ == "__main__":
    langs = sys.argv[1:] or ["EN", "CN"]
    ok = all(build(l) for l in langs)
    ok = all([build_single(n) for n in SINGLES]) and ok
    sys.exit(0 if ok else 1)
