"""Locate the pandoc binary without assuming it is on PATH.

Homebrew installs to /opt/homebrew/bin, which a login shell adds to PATH only if
the shell profile runs `brew shellenv`. It does here for one shell and not for
another, so a build script that just calls "pandoc" works for whoever set it up
and fails for everyone else with a bare FileNotFoundError.
"""
import os
import shutil
from pathlib import Path

CANDIDATES = [
    "/opt/homebrew/bin/pandoc",      # Homebrew on Apple silicon
    "/usr/local/bin/pandoc",         # Homebrew on Intel, and manual installs
    "/opt/local/bin/pandoc",         # MacPorts
    str(Path.home() / ".local/bin/pandoc"),
]


def pandoc():
    """Absolute path to pandoc. Raises with something actionable if absent."""
    env = os.environ.get("PANDOC")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    found = shutil.which("pandoc")
    if found:
        return found
    for c in CANDIDATES:
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise SystemExit(
        "找不到 pandoc。\n"
        "  已找过 PATH 与：\n    " + "\n    ".join(CANDIDATES) + "\n"
        "  装了但不在 PATH 上，可以这样指定：\n"
        "    PANDOC=/opt/homebrew/bin/pandoc .venv/bin/python 07_paper2_eicu/48_build_submission_docx.py\n"
        "  或把它永久加进 PATH：\n"
        "    echo 'eval \"$(/opt/homebrew/bin/brew shellenv)\"' >> ~/.zprofile\n"
        "  没装的话：brew install pandoc")


if __name__ == "__main__":
    print(pandoc())
