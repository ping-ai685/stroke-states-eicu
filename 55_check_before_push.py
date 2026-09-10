"""
Paper 2: the last check before anything reaches the public repository.

This deliberately shares no code with 44_repository_audit.py. 44 decides what may
be published by classifying files; 45 re-runs that same classification on what it
copied and calls it an independent audit, which it is not -- a wrong rule is wrong
in both places. That is not hypothetical: 44 classified every .py as analysis code,
so the scripts that build the private review notebooks were being published, and
later the scripts that build the operations manual were copied into the repository
for the same reason. Naming the directory fixed the rule. It did not fix the fact
that the only thing checking the rule was the rule.

So this asks a different question, of the artefact rather than of the decision:
given a directory that is about to become public, is there anything in it that
must never be public? It works from the file tree and from file contents. It knows
nothing about verdicts, manifests, or why any file is there.

Three things end the check immediately:

  1. a path in the private tier (notebooks, the operations manual, anything hidden)
  2. a file type that cannot be published (notebooks, model binaries, PDFs)
  3. a patient identifier column in a CSV, or a machine-specific absolute path

Usage:  python 55_check_before_push.py [目标目录]
        默认检查 ../paper2_code_repository
Exit code 1 if anything is found, so it can be used as a git pre-push hook.
"""
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT = HERE.parent / "paper2_code_repository"

# 私有层：整个目录都不许公开，不论里面是什么类型的文件
PRIVATE_DIRS = {"notebooks", "manual", "review", "review_returned",
                "submission_JAMIA", "submission_medRxiv",
                "protocol_v1.0", "protocol_v1.1", "protocol_v1.2"}

# 不该出现在公开代码仓库里的文件类型
BANNED_EXT = {".ipynb", ".pkl", ".joblib", ".npz", ".npy", ".pt", ".pth",
              ".h5", ".parquet", ".feather", ".docx", ".pdf", ".env"}

# 病人级标识列。出现在 CSV 表头就是病人级数据，不论文件叫什么。
ID_COLS = {"subject_id", "hadm_id", "stay_id", "icustay_id",
           "patientunitstayid", "patienthealthsystemstayid", "uniquepid"}

# 本机特定的绝对路径。发出去别人跑不了，而且暴露目录结构。
MACHINE = re.compile(r"/Users/[A-Za-z0-9_.-]+/|/Volumes/[A-Za-z0-9_. -]+/")


def scan(root):
    problems = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = rel.parts

        if parts and parts[0] == ".git":
            continue                      # 版本库自身，不是内容

        if any(seg.startswith(".") for seg in parts[:-1]):
            problems.append((rel, "隐藏目录"))
            continue
        if parts[0] in PRIVATE_DIRS:
            problems.append((rel, f"私有层目录 {parts[0]}/"))
            continue
        if p.suffix.lower() in BANNED_EXT:
            problems.append((rel, f"不可公开的文件类型 {p.suffix}"))
            continue

        if p.suffix.lower() == ".csv":
            try:
                with p.open(encoding="utf-8", newline="") as fh:
                    header = next(csv.reader(fh), [])
            except (UnicodeDecodeError, StopIteration):
                header = []
            hit = ID_COLS & {h.strip().lower() for h in header}
            if hit:
                problems.append((rel, f"病人标识列 {sorted(hit)}"))
                continue

        # data_paths.py 的工作就是持有那个路径，并且在文档里告诉读者怎么覆盖它。
        # 例外按确切文件名给出，不用通配——一条模糊的例外会把整条规则蛀空。
        if rel.as_posix() == "data_paths.py":
            continue

        if p.suffix.lower() in {".py", ".md", ".txt", ".yml", ".yaml", ".json"}:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in MACHINE.finditer(text):
                line = text[:m.start()].count("\n") + 1
                problems.append((rel, f"第 {line} 行有本机绝对路径 {m.group(0)}"))
                break
    return problems


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not root.exists():
        raise SystemExit(f"找不到目录：{root}")

    files = [p for p in root.rglob("*")
             if p.is_file() and ".git" not in p.relative_to(root).parts]
    problems = scan(root)

    print(f"\n检查 {root.name}/ —— {len(files)} 个文件\n")
    if not problems:
        print("  ✅ 没有发现私有层文件、病人标识列或本机路径")
        print("\n这一条与 44 号的分类规则无关：它只看目录里实际有什么。")
        return
    print(f"  ❌ {len(problems)} 处不该公开的内容：\n")
    for rel, why in problems:
        print(f"     {rel}")
        print(f"        {why}")
    print("\n先处理掉再推送。")
    sys.exit(1)


if __name__ == "__main__":
    main()
