"""
Paper 2: build the pandoc reference document that sets the submission's page layout.

Two defects motivated writing this down instead of leaving it as a shell command.

First, pandoc's own reference.docx carries no page size and no margins at all, so
Word falls back to whatever the machine's locale supplies -- Letter on one, A4 on
another. A manuscript whose layout depends on who opens it is not a layout.

Second, JAMIA requires double spacing, and the obvious way to apply it (rewrite
every <w:spacing> in styles.xml) also rewrites `Compact`, the style pandoc puts
inside every table cell. That silently double-spaced all five tables and made them
tall and cramped. Double spacing belongs to running text, not to table cells.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pandoc_path import pandoc   # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "manuscript/jamia_reference.docx"
TMP = HERE / "manuscript/.pandoc_default_reference.docx"

# Double-spaced: the styles that carry running text.
DOUBLE = {"Normal", "BodyText", "FirstParagraph", "Abstract", "BlockText",
          "Definition", "Bibliography"}
# Single-spaced and smaller: table cells, captions, footnotes.
TIGHT = {"Compact": 18, "Caption": 18, "TableCaption": 18, "ImageCaption": 18,
         "FootnoteText": 18, "FootnoteBlockText": 18}

PGSZ = '<w:pgSz w:w="12240" w:h="15840"/>'          # US Letter, portrait
PGMAR = ('<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
         'w:header="720" w:footer="720" w:gutter="0"/>')


def restyle(styles):
    out = []
    for block in re.split(r"(?=<w:style )", styles):
        m = re.search(r'w:styleId="([^"]+)"', block)
        sid = m.group(1) if m else None
        if sid in DOUBLE:
            block = re.sub(r"<w:spacing[^/]*?/>",
                           '<w:spacing w:line="480" w:lineRule="auto" '
                           'w:before="0" w:after="240"/>', block)
            if "<w:spacing" not in block and "<w:pPr>" in block:
                block = block.replace("<w:pPr>",
                                      '<w:pPr><w:spacing w:line="480" '
                                      'w:lineRule="auto" w:before="0" w:after="240"/>', 1)
        elif sid in TIGHT:
            block = re.sub(r"<w:spacing[^/]*?/>",
                           '<w:spacing w:line="240" w:lineRule="auto" '
                           'w:before="40" w:after="40"/>', block)
            sz = TIGHT[sid]
            if re.search(r"<w:sz w:val=\"\\d+\"\\s*/>", block):
                block = re.sub(r'<w:sz w:val="\d+"\s*/>', f'<w:sz w:val="{sz}"/>', block)
            elif "<w:rPr>" in block:
                block = block.replace("<w:rPr>", f'<w:rPr><w:sz w:val="{sz}"/>', 1)
        out.append(block)
    styles = "".join(out)
    # Times New Roman everywhere; Consolas is for code, and pandoc leaks it into
    # the document defaults.
    styles = re.sub(r'w:ascii="Consolas" w:hAnsi="Consolas"',
                    'w:ascii="Times New Roman" w:hAnsi="Times New Roman"',
                    styles, count=1)
    styles = styles.replace(
        '<w:rFonts w:asciiTheme="minorHAnsi" w:hAnsiTheme="minorHAnsi"/>',
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>')
    return styles


def set_page(document):
    if "<w:pgSz" in document:
        document = re.sub(r"<w:pgSz[^>]*/>", PGSZ, document)
    else:
        document = document.replace("<w:sectPr>", "<w:sectPr>" + PGSZ, 1)
    if "<w:pgMar" in document:
        document = re.sub(r"<w:pgMar[^>]*/>", PGMAR, document)
    else:
        document = document.replace("<w:sectPr>", "<w:sectPr>" + PGMAR, 1)
    return document


def main():
    with open(TMP, "wb") as fh:
        subprocess.run([pandoc(), "--print-default-data-file", "reference.docx"],
                       stdout=fh, check=True)
    zin = zipfile.ZipFile(TMP)
    styles = restyle(zin.read("word/styles.xml").decode())
    document = set_page(zin.read("word/document.xml").decode())

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in zin.infolist():
            if item.filename == "word/styles.xml":
                zo.writestr(item, styles)
            elif item.filename == "word/document.xml":
                zo.writestr(item, document)
            else:
                zo.writestr(item, zin.read(item.filename))
    zin.close()
    TMP.unlink()

    z = zipfile.ZipFile(OUT)
    s, d = z.read("word/styles.xml").decode(), z.read("word/document.xml").decode()
    def spacing_of(sid):
        m = re.search(r'<w:style [^>]*w:styleId="' + sid + r'".*?</w:style>', s, re.S)
        v = re.findall(r'<w:spacing[^>]*w:line="(\d+)"', m.group(0)) if m else []
        return v[0] if v else "-"
    print(f"wrote {OUT.name}")
    print(f"  页面     {re.search(r'<w:pgSz[^>]*/>', d).group(0)}")
    print(f"  页边距   {re.search(r'<w:pgMar[^>]*/>', d).group(0)[:70]}…")
    print(f"  正文行距 BodyText={spacing_of('BodyText')} Normal={spacing_of('Normal')} "
          f"(480 = 双倍)")
    print(f"  表格行距 Compact={spacing_of('Compact')} (240 = 单倍)")
    print(f"  Consolas 残留 {s.count('Consolas')} 处")


if __name__ == "__main__":
    main()
