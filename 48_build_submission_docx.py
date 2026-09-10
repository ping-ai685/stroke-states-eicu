"""
Paper 2: assemble the JAMIA submission .docx from its section files.

The first build was a pandoc command typed at a shell. That is fine once and a
liability afterwards: nothing records which section files went in, in which
order, so a rebuild months later can silently differ from what was submitted.
This script fixes the order in code, rebuilds both languages, and then verifies
that what came out contains what went in.
"""
import re
import unicodedata
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


CELL_PAD = 60            # side padding inside each cell, per side
MIN_COL = 420
MAX_UNBREAKABLE = 18     # beyond this a token is an identifier, not a word
BOLD = 1.08              # the header row is bold and therefore wider
# The character width is an estimate. "-0.061" was given 634 twips against an
# estimated need of 630, and Word resolved the four-twip shortfall by breaking
# after the hyphen -- putting the minus sign on its own line, where a reader can
# lose it. Estimates that tight are not estimates.
SAFETY = 1.12
SIZES = (18, 17, 16)     # half-points: try 9pt, then 8.5, then 8


def _char_twips(half_points):
    """Width of an average Times New Roman character, in twips."""
    return half_points * 5


def disp_len(text):
    """Character count in *display* widths, not code points.

    A CJK character occupies about twice the width of an average Latin one at the
    same point size, so counting code points starves any column whose header or
    content is Chinese. In the Chinese manuscript and in the operations manual this
    showed up as a five-character header wrapped onto three lines beside a column of
    short numbers that had been given four inches.
    """
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in text)


def fit_tables(path, avail=9360):
    """Give each column the width its content needs, padding included.

    The first version of this computed the text width and stopped there, which is
    why words still broke: every column also spends its side padding, and nine
    columns of it came to 1.35 inches that the arithmetic never saw. Table 2 needed
    6.97 inches of a 6.50-inch page and Word resolved the shortfall by breaking
    "Windows" into "Windo / ws" and "-0.061" into "-0.06 / 1".

    So padding is counted, the padding itself is narrower than Word's default, and
    if a table still does not fit its type is stepped down a half point at a time.
    Only when even the smallest size fails is anything allowed to break.
    """
    import shutil, tempfile
    z = zipfile.ZipFile(path)
    doc = z.read("word/document.xml").decode()
    members = {n: z.read(n) for n in z.namelist()}
    z.close()
    RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
    report = []

    def one(m):
        tbl = m.group(0)
        rows = re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S)
        if not rows:
            return tbl
        grid = [re.findall(r"<w:tc>.*?</w:tc>", r, re.S) for r in rows]
        ncols = max(len(g) for g in grid)

        longest, bulk = [], []
        for c in range(ncols):
            lw, tot = 1, 0
            for r_i, g in enumerate(grid):
                if c >= len(g):
                    continue
                txt = " ".join("".join(RUN.findall(g[c])).split())
                tot += disp_len(txt)
                f = BOLD if r_i == 0 else 1.0
                for tok in txt.split():
                    lw = max(lw, disp_len(tok) * f)
            longest.append(min(lw, MAX_UNBREAKABLE * BOLD))
            bulk.append(tot / max(1, len(grid)))

        size = SIZES[-1]
        for sz in SIZES:
            ch = _char_twips(sz)
            base = [max(MIN_COL, int(lw * ch * SAFETY) + 2 * CELL_PAD) for lw in longest]
            if sum(base) <= avail:
                size = sz
                break
        ch = _char_twips(size)
        base = [max(MIN_COL, int(lw * ch * SAFETY) + 2 * CELL_PAD) for lw in longest]
        spare = avail - sum(base)
        if spare < 0:
            k = avail / sum(base)
            widths = [int(b * k) for b in base]
        else:
            demand = [max(0.0, bulk[c] * ch - base[c]) for c in range(ncols)]
            tot_d = sum(demand)
            widths = [int(base[c] + (spare * demand[c] / tot_d if tot_d else spare / ncols))
                      for c in range(ncols)]
        widths[-1] += avail - sum(widths)
        report.append((ncols, size, sum(base) <= avail))

        tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>",
                     "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
                     + "</w:tblGrid>", tbl, flags=re.S)
        ins = (f'<w:tblW w:w="{avail}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
               f'<w:tblCellMar><w:top w:w="40" w:type="dxa"/>'
               f'<w:left w:w="{CELL_PAD}" w:type="dxa"/>'
               f'<w:bottom w:w="40" w:type="dxa"/>'
               f'<w:right w:w="{CELL_PAD}" w:type="dxa"/></w:tblCellMar>')
        tbl = re.sub(r"<w:tblW[^>]*/>", "", tbl)
        tbl = re.sub(r"<w:tblLayout[^>]*/>", "", tbl)
        tbl = re.sub(r"<w:tblCellMar>.*?</w:tblCellMar>", "", tbl, flags=re.S)
        if "<w:tblPr>" in tbl:
            tbl = tbl.replace("<w:tblPr>", "<w:tblPr>" + ins, 1)
        elif "<w:tblPr/>" in tbl:
            tbl = tbl.replace("<w:tblPr/>", "<w:tblPr>" + ins + "</w:tblPr>", 1)

        for r_i, row in enumerate(rows):
            new = row
            props = "<w:cantSplit/>" + ("<w:tblHeader/>" if r_i == 0 else "")
            if "<w:trPr>" in new:
                new = new.replace("<w:trPr>", "<w:trPr>" + props, 1)
            else:
                new = re.sub(r"(<w:tr[^>]*>)", r"\1<w:trPr>" + props + "</w:trPr>", new, count=1)
            for c_i, cell in enumerate(re.findall(r"<w:tc>.*?</w:tc>", new, re.S)):
                if c_i >= ncols:
                    continue
                w = f'<w:tcW w:w="{widths[c_i]}" w:type="dxa"/>'
                fixed = (cell.replace("<w:tcPr>", "<w:tcPr>" + w, 1) if "<w:tcPr>" in cell
                         else cell.replace("<w:tcPr />", "<w:tcPr>" + w + "</w:tcPr>", 1))
                new = new.replace(cell, fixed, 1)
            # Stamp the chosen size on every run in the table.
            new = re.sub(r"<w:sz w:val=\"\d+\"\s*/>", f'<w:sz w:val="{size}"/>', new)
            new = re.sub(r"(<w:rPr>)(?!<w:sz)", f'\\1<w:sz w:val="{size}"/>', new)
            new = re.sub(r"(<w:r>)(?!<w:rPr>)", f'\\1<w:rPr><w:sz w:val="{size}"/></w:rPr>', new)
            tbl = tbl.replace(row, new, 1)
        return tbl

    doc = re.sub(r"<w:tbl>.*?</w:tbl>", one, doc, flags=re.S)
    members["word/document.xml"] = doc.encode()
    tmp = tempfile.mktemp(suffix=".docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, data in members.items():
            zo.writestr(name, data)
    shutil.move(tmp, path)
    return report


_NUM_CHARS = set("0123456789.,+-%()[]/ \u2212\u2013\u2014\uff08\uff09\uff0c\uff05"
                 "\u00b1<>=~")
_NUM_WORDS = ("to", "\u81f3")


def is_numeric(text):
    """Whether a cell holds a number rather than words.

    Matching a pattern kept failing on the forms these tables use: full-width
    parentheses in the Chinese version, "27.8 / 39.2", "+0.057 (+0.052 to +0.062)".
    Strip everything that can legitimately appear inside a number and see whether
    anything is left. "6 h" keeps its word and stays left-aligned, which is right:
    it is a label, not a value.
    """
    t = text.strip()
    if not t or not any(c.isdigit() for c in t):
        return False
    for w in _NUM_WORDS:
        t = t.replace(w, "")
    return not (set(t) - _NUM_CHARS)


def align_numeric_columns(path):
    """Centre the columns that hold numbers; leave text columns left.

    Deciding per column rather than per cell keeps a column's header with its
    values.
    """
    import shutil, tempfile
    z = zipfile.ZipFile(path)
    doc = z.read("word/document.xml").decode()
    members = {n: z.read(n) for n in z.namelist()}
    z.close()
    RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
    n_col = 0

    def one_table(m):
        nonlocal n_col
        tbl = m.group(0)
        rows = re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S)
        if len(rows) < 2:
            return tbl
        grid = [re.findall(r"<w:tc>.*?</w:tc>", r, re.S) for r in rows]
        ncols = max(len(g) for g in grid)
        numeric = []
        for c in range(ncols):
            vals = ["".join(RUN.findall(g[c])).strip()
                    for g in grid[1:] if c < len(g) and "".join(RUN.findall(g[c])).strip()]
            numeric.append(bool(vals) and
                           sum(is_numeric(v) for v in vals) / len(vals) >= 0.6)
        n_col += sum(numeric)
        for row, cells in zip(rows, grid):
            new_row = row
            for c_i, cell in enumerate(cells):
                if c_i >= ncols or not numeric[c_i]:
                    continue
                fixed = cell.replace("<w:pPr>", '<w:pPr><w:jc w:val="center"/>')
                if fixed == cell:
                    fixed = cell.replace("<w:p>", '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>')
                new_row = new_row.replace(cell, fixed, 1)
            tbl = tbl.replace(row, new_row, 1)
        return tbl

    doc = re.sub(r"<w:tbl>.*?</w:tbl>", one_table, doc, flags=re.S)
    members["word/document.xml"] = doc.encode()
    tmp = tempfile.mktemp(suffix=".docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, data in members.items():
            zo.writestr(name, data)
    shutil.move(tmp, path)
    return n_col


def true_minus(path):
    """Use a minus sign for negative numbers, not a hyphen.

    Word may break a line after a hyphen, and in a narrow column it does: the
    proof of this manuscript rendered -0.061 as "-" on one line and "0.061" on the
    next, where the sign is easily lost. U+2212 is the character a negative number
    should carry anyway, and Word does not break after it.
    """
    import shutil, tempfile
    z = zipfile.ZipFile(path)
    doc = z.read("word/document.xml").decode()
    members = {n: z.read(n) for n in z.namelist()}
    z.close()
    n = 0

    def fix(m):
        nonlocal n
        body = m.group(0)
        def one_run(r):
            nonlocal n
            txt = r.group(1)
            new = re.sub(r"(?<![\w)])-(?=\d)", "\u2212", txt)
            n += new != txt
            return r.group(0).replace(txt, new) if new != txt else r.group(0)
        return re.sub(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", one_run, body, flags=re.S)

    doc = re.sub(r"<w:tbl>.*?</w:tbl>", fix, doc, flags=re.S)
    members["word/document.xml"] = doc.encode()
    tmp = tempfile.mktemp(suffix=".docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, data in members.items():
            zo.writestr(name, data)
    shutil.move(tmp, path)
    return n


def hanging_references(path, heading):
    """Give the reference list a hanging indent.

    The references are plain markdown paragraphs, not a citeproc bibliography, so
    pandoc styles them as body text and "[12]" runs into the line above it.
    """
    import shutil, tempfile
    z = zipfile.ZipFile(path)
    doc = z.read("word/document.xml").decode()
    members = {n: z.read(n) for n in z.namelist()}
    z.close()

    paras = re.findall(r"<w:p[ >].*?</w:p>", doc, re.S)
    RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
    inside, n = False, 0
    for para in paras:
        text = "".join(RUN.findall(para)).strip()
        if text == heading:
            inside = True
            continue
        if inside:
            if '<w:pStyle w:val="Heading' in para:
                inside = False
                continue
            if not text:
                continue
            ind = '<w:ind w:left="567" w:hanging="567"/>'
            if "<w:pPr>" in para:
                new = para.replace("<w:pPr>", "<w:pPr>" + ind, 1)
            else:
                new = para.replace("<w:p>", "<w:p><w:pPr>" + ind + "</w:pPr>", 1)
            doc = doc.replace(para, new, 1)
            n += 1
    members["word/document.xml"] = doc.encode()
    tmp = tempfile.mktemp(suffix=".docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, data in members.items():
            zo.writestr(name, data)
    shutil.move(tmp, path)
    return n


def landscape(path):
    """Turn a document sideways.

    The supplementary tables run to ten columns, two of them carrying long text
    fields. In portrait each column gets about 0.65 inch, which no width
    calculation can rescue; landscape gives the text area 9.5 inches.
    """
    import shutil, tempfile
    z = zipfile.ZipFile(path)
    doc = z.read("word/document.xml").decode()
    members = {n: z.read(n) for n in z.namelist()}
    z.close()
    doc = re.sub(r"<w:pgSz[^>]*/>",
                 '<w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/>', doc)
    doc = re.sub(r"<w:pgMar[^>]*/>",
                 '<w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" '
                 'w:header="720" w:footer="720" w:gutter="0"/>', doc)
    members["word/document.xml"] = doc.encode()
    tmp = tempfile.mktemp(suffix=".docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, data in members.items():
            zo.writestr(name, data)
    shutil.move(tmp, path)


CAPTION = re.compile(r"^(Table \d+\.|Supplementary Table S\d+\.|表\s*\d+\.|补充表\s*S\d+\.)")


def style_table_captions(path):
    """Set the table captions in small single-spaced type, tight above the table.

    A caption inherits body text otherwise: 12pt, double-spaced, with a full blank
    line after it. That is three or four lines of large type sitting between the
    sentence that cites the table and the table itself, and it pushes the table
    onto the next page as often as not. Journals set captions smaller, single
    spaced, and glued to the table below.
    """
    import shutil, tempfile
    z = zipfile.ZipFile(path)
    doc = z.read("word/document.xml").decode()
    members = {n: z.read(n) for n in z.namelist()}
    z.close()
    RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
    n = 0
    for para in re.findall(r"<w:p[ >].*?</w:p>", doc, re.S):
        text = "".join(RUN.findall(para)).strip()
        if not CAPTION.match(text):
            continue
        props = ('<w:spacing w:line="240" w:lineRule="auto" w:before="120" w:after="60"/>'
                 '<w:keepNext/>')          # never orphan a caption from its table
        if "<w:pPr>" in para:
            new = para.replace("<w:pPr>", "<w:pPr>" + props, 1)
        else:
            new = para.replace("<w:p>", "<w:p><w:pPr>" + props + "</w:pPr>", 1)
        # 10pt, to match the table body rather than the running text
        new = re.sub(r"(<w:r>)(?!<w:rPr>)", r'\1<w:rPr><w:sz w:val="20"/></w:rPr>', new)
        new = re.sub(r"<w:rPr>(?!<w:sz)", '<w:rPr><w:sz w:val="20"/>', new)
        doc = doc.replace(para, new, 1)
        n += 1
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
    rep = fit_tables(out)
    n_tbl = len(rep)
    n_num = align_numeric_columns(out)
    n_cap = style_table_captions(out)
    n_min = true_minus(out)
    n_ref = hanging_references(out, "References" if lang == "EN" else "参考文献")

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
    print(f"     ✅ {n_tbl} 张表满宽自适应，{n_num} 个数字列居中，{n_ref} 条文献悬挂缩进")
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
    wide = name.startswith("supplementary")
    # Landscape must be decided before the widths are computed: the text area is
    # 13680 twips sideways, 9360 upright, and a table laid out for one looks wrong
    # in the other.
    if wide:
        landscape(out)
    n = len(fit_tables(out, avail=13680 if wide else 9360))
    align_numeric_columns(out)
    style_table_captions(out)
    true_minus(out)
    print(f"  {out.name}  {out.stat().st_size:,} bytes  {n} 张表已设为满宽自适应"
          + ("，横排 9 英寸" if wide else ""))
    return True


if __name__ == "__main__":
    langs = sys.argv[1:] or ["EN", "CN"]
    ok = all(build(l) for l in langs)
    ok = all([build_single(n) for n in SINGLES]) and ok
    sys.exit(0 if ok else 1)
