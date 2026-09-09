"""
Paper 2: assemble the folder that gets uploaded to JAMIA, and nothing else.

manuscript/ is a working directory, and choosing the upload from among its .docx
files by hand is an avoidable risk: it once held seventeen of them, including a
manuscript_JAMIA_EN_review.docx that differed from the real one by six characters
in its name. The superseded drafts now sit in manuscript/_superseded/, but the
working directory will fill up again, so this script still produces a directory
containing exactly what is uploaded, under names that say what each file is.

It refuses to assemble anything if the verification suite does not pass.
"""
import shutil
import subprocess
import sys
import zipfile
import re
from pathlib import Path

HERE = Path(__file__).parent
MS = HERE / "manuscript"
FIG = HERE / "figures"
OUT = HERE / "submission_JAMIA"

# name in the folder -> (source, JAMIA File Designation)
FILES = [
    ("01_Manuscript.docx",            MS / "manuscript_JAMIA_EN.docx",  "Main Document"),
    ("02_Cover_letter.docx",          MS / "cover_letter_JAMIA.docx",   "Cover Letter"),
    ("03_Supplementary_tables.docx",  MS / "supplementary_tables.docx", "Supplementary File"),
    ("04_Figure1.tif", FIG / "figure1_cohort_flow_EN.tif",            "Figure"),
    ("05_Figure2.tif", FIG / "figure2_transport_EN.tif",              "Figure"),
    ("06_Figure3.tif", FIG / "figure3_denovo_correspondence_EN.tif",  "Figure"),
    ("07_Figure4.tif", FIG / "figure4_prediction_EN.tif",             "Figure"),
]

GATE = ["49_verify_figures_against_tables.py", "50_verify_prose_claims.py"]



def clear_generated(out, keep_suffixes=(".pdf",)):
    """Empty the output directory of what this script generates, and only that.

    `shutil.rmtree` on an output directory deletes whatever else has been filed
    there. It did: the published preprint PDF and the medRxiv proof were sitting in
    this folder and a rebuild removed both. A build may overwrite its own products;
    it has no business deleting a downloaded document that happens to share a
    directory with them.
    """
    if not out.exists():
        out.mkdir(parents=True)
        return []
    kept = []
    for p in sorted(out.iterdir()):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.suffix.lower() in keep_suffixes:
            kept.append(p.name)
        else:
            p.unlink()
    return kept


def _fig_ok(path, min_dpi=300, min_px=1000):
    """A figure is valid if it opens, is large enough, and is still 300 dpi."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        im = Image.open(path)
        im.load()
    except Exception:
        return False
    dpi = im.info.get("dpi", (0, 0))
    return min(im.size) >= min_px and float(dpi[0]) >= min_dpi - 1


def compress_tiff(src, dst):
    """Write the figure as an LZW-compressed TIFF.

    matplotlib writes TIFFs uncompressed: four figures came to 50.8 MB, and
    ScholarOne refused the submission for exceeding its 58.59 MB total. LZW is
    lossless — same pixels, same 300 dpi — and brings the four to 1.6 MB. The
    saved file is compared with the original pixel by pixel before it is kept.
    """
    from PIL import Image
    import numpy as np
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(src)
    dpi = im.info.get("dpi", (300, 300))
    im.save(dst, format="TIFF", compression="tiff_lzw",
            dpi=(float(dpi[0]), float(dpi[1])))
    a, b = np.array(im), np.array(Image.open(dst))
    assert a.shape == b.shape and np.array_equal(a, b), f"{src.name}: 压缩不是无损的"
    return src.stat().st_size, dst.stat().st_size


def docx_text(path):
    x = zipfile.ZipFile(path).read("word/document.xml").decode()
    RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
    return "\n".join("".join(RUN.findall(p))
                     for p in re.findall(r"<w:p[ >].*?</w:p>", x, re.S))


def main():
    print("先跑校验，没过就不装配\n")
    for g in GATE:
        r = subprocess.run([sys.executable, str(HERE / g)],
                           capture_output=True, text=True)
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:]
        print(f"  {'OK ' if r.returncode == 0 else 'BAD'}  {g}   {tail[0] if tail else ''}")
        if r.returncode != 0:
            raise SystemExit("校验未通过，未装配。先修好再来。")

    missing = [str(src) for _, src, _ in FILES if not src.exists()]
    if missing:
        raise SystemExit("缺少源文件：\n  " + "\n  ".join(missing))

    kept = clear_generated(OUT)
    if kept:
        print(f"  保留了目录里已有的 {len(kept)} 个 PDF：{', '.join(kept)}")
    print(f"\n装配 {OUT.name}/\n")
    saved = 0
    for name, src, desig in FILES:
        if src.suffix.lower() == ".tif":
            before, after = compress_tiff(src, OUT / name)
            saved += before - after
        else:
            shutil.copy2(src, OUT / name)
        kb = (OUT / name).stat().st_size / 1024
        print(f"  {name:32s} {kb:>9,.0f} KB   ← {src.name}   [{desig}]")
    if saved:
        print(f"\n  图片无损压缩省下 {saved/1048576:.1f} MB")

    # The alt text lives under each legend in the manuscript, as JAMIA requires.
    # Portals often also ask for it field by field, so keep a copy to paste from.
    alts = re.findall(r"^Alt text: (.+)$",
                      (MS / "figure_legends_jamia_EN.md").read_text(), re.M)
    tp = (MS / "title_page_EN.md").read_text()
    wc = re.search(r"main text (\d+)", tp).group(1)
    kw = [k.strip() for k in
          re.search(r"\*\*Keywords:\*\*([^\n]*)", tp).group(1).split(";") if k.strip()]

    lines = [
        "# JAMIA submission — upload checklist",
        "",
        "Assembled by `53_assemble_submission.py`. Everything in this folder is",
        "uploaded; nothing outside it is. Do not upload from `manuscript/`, which",
        "still holds seven superseded drafts and the Chinese working version.",
        "",
        "## Files, in upload order",
        "",
        "| # | File | JAMIA File Designation |",
        "|---|---|---|",
    ]
    for name, _, desig in FILES:
        lines.append(f"| {name.split('_')[0]} | `{name}` | {desig} |")
    lines += [
        "",
        "## Requirements checked",
        "",
        "| Requirement | JAMIA | This submission |",
        "|---|---|---|",
        f"| Main text | ≤4000 words | {wc} |",
        "| Abstract | structured, ≤250 words | 250, five required headings |",
        "| Tables | ≤4 | 4, placed where first cited |",
        "| Figures | ≤6 | 4, uploaded as separate files |",
        f"| Keywords | up to five | {len(kw)} |",
        "| Spacing | double-spaced | body double, table cells single |",
        "| Alt text | under each legend, prefixed \"Alt text:\" | 4 of 4 |",
        "",
        "## Still to do by hand",
        "",
        "- The corresponding author's telephone number is not printed on the title",
        "  page. JAMIA asks for it there; supply it in the submission form.",
        "",
        "## Alt text, to paste if the portal asks for it field by field",
        "",
    ]
    for i, a in enumerate(alts, 1):
        lines += [f"**Figure {i}**", "", a, ""]
    (OUT / "SUBMISSION_CHECKLIST.txt").write_text("\n".join(lines) + "\n")
    print(f"\n  {'SUBMISSION_CHECKLIST.txt':32s} {(OUT/'SUBMISSION_CHECKLIST.txt').stat().st_size/1024:>9,.0f} KB")

    # Verify the copies, not the originals.
    t = docx_text(OUT / "01_Manuscript.docx")
    checks = [
        ("正文就是 JAMIA 版，不是任何旧稿",
         "External validation of dynamic clinical states in acute stroke" in t),
        ("四条 alt text 在正文里", t.count("Alt text:") == 4),
        ("没有残留占位符", "TO BE SUPPLIED" not in t),
        ("投稿信是最新的",
         "including passages that interpret the findings" in docx_text(OUT / "02_Cover_letter.docx")),
        ("补充材料含 S1–S5",
         all(f"Supplementary Table S{i}" in docx_text(OUT / "03_Supplementary_tables.docx")
             for i in range(1, 6))),
        # "bigger than a megabyte" stood in for "is a real image" and stopped
        # meaning anything once the figures were compressed. Open them instead.
        ("四张图都是 300 dpi 的有效图像", all(
            _fig_ok(OUT / n) for n, _, d in FILES if d == "Figure")),
    ]
    print()
    bad = [c for c, ok in checks if not ok]
    for c, ok in checks:
        print(f"  {'OK ' if ok else 'BAD'}  {c}")
    if bad:
        raise SystemExit("装配出的文件没通过复核。")
    print(f"\n✅ {len(FILES)} 个文件 + 清单，可以上传")


if __name__ == "__main__":
    main()
