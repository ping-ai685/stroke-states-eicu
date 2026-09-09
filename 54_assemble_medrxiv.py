"""
Paper 2: assemble the medRxiv preprint package.

medRxiv differs from JAMIA in three ways that matter here. It accepts one
self-contained file, which is what a preprint reader wants: the figures should be
in the document, not in seven separate uploads. It requires an ethics declaration
at submission. And it has no word limit, so nothing needs trimming.

The text is the same text: this script reads the same section files the JAMIA
build reads, so the preprint and the submission cannot drift apart.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from importlib import import_module
from pandoc_path import pandoc

_b48 = import_module("48_build_submission_docx")
merge_results = import_module("51_merge_results_with_tables").merge

MS = HERE / "manuscript"
FIG = HERE / "figures"
OUT = HERE / "submission_medRxiv"

ORDER = ["title_page_EN.md", "abstract_jamia_EN.md", "intro_jamia_EN.md",
         "methods_jamia_EN.md", "results_jamia_with_tables_EN.md",
         "discussion_jamia_EN.md", "data_availability_EN.md",
         "declarations_EN.md", "references_jamia_EN.md"]

PNG = {1: "figure1_cohort_flow_EN.png", 2: "figure2_transport_EN.png",
       3: "figure3_denovo_correspondence_EN.png", 4: "figure4_prediction_EN.png"}



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


def figures_with_images():
    """Legends, alt text and the images themselves, in one file."""
    src = (MS / "figure_legends_jamia_EN.md").read_text()
    out = []
    for blk in src.split("\n\n"):
        blk = blk.strip()
        if not blk:
            continue
        out.append(blk)
        m = re.match(r"\*\*Figure (\d)\.", blk)
        if m:
            continue                      # image goes after the alt text, not the legend
        m2 = re.match(r"Alt text:", blk)
        if m2 and out:
            # find which figure this alt text belongs to
            for prev in reversed(out[:-1]):
                mm = re.match(r"\*\*Figure (\d)\.", prev)
                if mm:
                    p = FIG / PNG[int(mm.group(1))]
                    assert p.exists(), f"缺少 {p}"
                    out.append(f"![]({p.as_posix()})")
                    break
    dst = MS / ".medrxiv_figures_EN.md"
    dst.write_text("\n\n".join(out) + "\n")
    return dst


def main():
    for g in ["49_verify_figures_against_tables.py", "50_verify_prose_claims.py"]:
        r = subprocess.run([sys.executable, str(HERE / g)], capture_output=True, text=True)
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:]
        print(f"  {'OK ' if r.returncode == 0 else 'BAD'}  {g}   {tail[0] if tail else ''}")
        if r.returncode:
            raise SystemExit("校验未通过，未装配。")

    assert merge_results("EN"), "表格合并失败"
    figs = figures_with_images()
    files = [MS / f for f in ORDER] + [figs]
    missing = [f.name for f in files if not f.exists()]
    assert not missing, f"缺少 {missing}"

    kept = clear_generated(OUT)
    if kept:
        print(f"  保留了目录里已有的 {len(kept)} 个 PDF：{', '.join(kept)}")
    doc = OUT / "medRxiv_preprint.docx"
    subprocess.run([pandoc(), *[str(f) for f in files],
                    f"--reference-doc={MS / 'jamia_reference.docx'}",
                    "--resource-path", str(HERE), "-o", str(doc)], check=True)
    # Every step the JAMIA build applies, in the same order. Listing them here by
    # hand is how the minus-sign fix reached the submission and not the preprint;
    # taking them from one place stops the two drifting again.
    n_tbl = _b48.fit_tables(doc)
    _b48.align_numeric_columns(doc)
    _b48.style_table_captions(doc)
    n_min = _b48.true_minus(doc)
    n_ref = _b48.hanging_references(doc, "References")
    figs.unlink()

    z = zipfile.ZipFile(doc)
    media = [n for n in z.namelist() if n.startswith("word/media/")]
    t = _b48.docx_text(doc)
    checks = [
        ("四张图已内嵌", len(media) == 4),
        ("伦理声明在文内", "Institutional Review Board of Beth Israel" in t),
        ("资助声明在文内", "received no specific grant" in t),
        ("利益冲突在文内", "declare no competing interests" in t),
        ("数据可获得性在文内", "Data availability" in t),
        ("四条 alt text 在文内", t.count("Alt text:") == 4),
        ("正文是 JAMIA 同一份文本",
         "External validation of dynamic clinical states in acute stroke" in t),
    ]
    print(f"\n  {doc.name}  {doc.stat().st_size/1024/1024:.1f} MB  "
          f"{n_tbl} 张表自适应，{n_ref} 条文献悬挂缩进\n")
    for c, ok in checks:
        print(f"  {'OK ' if ok else 'BAD'}  {c}")
    if any(not ok for _, ok in checks):
        raise SystemExit("装配出的文件没通过复核。")

    (OUT / "MEDRXIV_CHECKLIST.txt").write_text("""# medRxiv preprint — submission notes

One file is uploaded: `medRxiv_preprint.docx`. The four figures are embedded in
it, after their legends and alt text, so nothing else needs to go with it.

The text is identical to the JAMIA submission. Both are built from the same
section files, so they cannot drift apart.

## The four declarations medRxiv asks for at submission

All four are already in the document; these are the answers for the web form.

**Ethics.** Retrospective analysis of two de-identified databases distributed
through PhysioNet under credentialed access. MIMIC-IV was collected under a waiver
of informed consent granted by the Institutional Review Board of Beth Israel
Deaconess Medical Center. eICU-CRD is de-identified to HIPAA Safe Harbor,
independently certified (Privacert, HIPAA certification no. 1031219-2). Secondary
analysis of existing de-identified public data; no additional IRB approval or
informed consent was required.

**Funding.** No specific grant from any funding agency in the public, commercial,
or not-for-profit sectors.

**Competing interests.** The authors declare no competing interests. No author or
institution received payments or services from a third party for any aspect of
this work in the past 36 months.

**Data availability.** MIMIC-IV v3.1 and eICU-CRD v2.0 are held at PhysioNet under
credentialed access and cannot be redistributed by the authors. Analysis code:
https://github.com/ping-ai685/stroke-states-eicu

## Order of operations

Post the preprint at the same time as the JAMIA submission, or one to two days
before it — not after. JAMIA permits preprints before acceptance and asks that
after acceptance the preprint is not replaced with the published version, only
linked to it.

## Subject area

Health Informatics.
""")
    print(f"\n✅ {OUT.name}/ 已装配（1 个文件 + 提交说明）")


if __name__ == "__main__":
    main()
