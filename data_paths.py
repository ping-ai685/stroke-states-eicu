"""Locate the two databases without hard-coding one machine's disk.

Ten scripts carried the same line:

    B = "/Volumes/Lexar/research data/eICU dataset/eicu-...-2.0"

It is the path on the machine this analysis was run on. Published as-is it fails
for every reader, and the README said to edit two files when there were ten. The
path also has to change whenever the drive is remounted under a different name,
which means editing ten files to fix one thing.

Set the environment variable once instead:

    export EICU_DIR="/path/to/eicu-collaborative-research-database-2.0"
    export MIMIC_DIR="/path/to/mimic-iv-3.1"

The directory each should point at is the one containing the .csv or .csv.gz
tables (patient.csv, diagnosis.csv for eICU; hosp/ and icu/ for MIMIC-IV).

Neither database may be redistributed. Both are at PhysioNet under credentialed
access; see the article's data availability statement.
"""
import os
from pathlib import Path

# 这台机器上的位置。留作默认值，找不到就报错说清楚该怎么办。
FALLBACKS = {
    "EICU_DIR": ["/Volumes/Lexar/research data/eICU dataset/"
                 "eicu-collaborative-research-database-2.0"],
    "MIMIC_DIR": ["/Volumes/Lexar/research data/MIMIC v_mac/MIMIC/"
                  "mimic-iv-3.1/mimic-iv-3.1"],
}

# 用来确认指对了目录的标志文件，各给几个候选，压缩与未压缩都算
MARKERS = {
    "EICU_DIR": ["patient.csv", "patient.csv.gz"],
    "MIMIC_DIR": ["hosp", "icu"],
}


def _resolve(var):
    tried = []
    env = os.environ.get(var)
    for cand in ([env] if env else []) + FALLBACKS[var]:
        p = Path(cand).expanduser()
        tried.append(str(p))
        if p.is_dir() and any((p / m).exists() for m in MARKERS[var]):
            return p
    raise SystemExit(
        f"\n找不到 {var} 指向的数据库目录。试过：\n"
        + "".join(f"  {t}\n" for t in tried)
        + f"\n把它设成你自己那份的路径，目录里应当能看到 "
        f"{' 或 '.join(MARKERS[var])}：\n"
        f"  export {var}=\"/your/path\"\n\n"
        "两个数据库都在 PhysioNet 的凭证访问下发布，不能由作者再分发。\n")


def eicu():
    """eICU-CRD v2.0 的目录。"""
    return _resolve("EICU_DIR")


def mimic():
    """MIMIC-IV v3.1 的目录。"""
    return _resolve("MIMIC_DIR")
