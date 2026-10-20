"""Read-only environment checks; no automatic installs or credential display."""
from __future__ import annotations

import shutil
import subprocess

from .data import child_environment
from .io import environment
from .models import StudySpec


def check(study=None):
    result = {"python": environment(), "rscript": shutil.which(study.rscript if study else "Rscript"),
              "r_packages": {}, "fonts": {}}
    from matplotlib import font_manager
    for name in ["Arial", "DejaVu Sans"]:
        try:
            result["fonts"][name] = font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            result["fonts"][name] = "unavailable"
    if result["rscript"]:
        script = ('for(p in c("jsonlite","limma","DESeq2")) '
                  'cat(p,"=",if(requireNamespace(p,quietly=TRUE))as.character(packageVersion(p))else"NOT_INSTALLED","\\n",sep="")')
        env = child_environment(study) if study else None
        try:
            response = subprocess.run([result["rscript"], "--vanilla", "-e", script], env=env, capture_output=True, text=True, timeout=90)
            result["r_packages"] = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)
        except subprocess.TimeoutExpired:
            result["r_packages"] = {"status": "timeout"}
    result["generated_code_ready"] = {"python": True, "r": bool(result["rscript"])}
    result["numerical_tools_ready"] = bool(result["r_packages"]) and all(
        result["r_packages"].get(p) not in {None, "NOT_INSTALLED"} for p in ["jsonlite", "limma", "DESeq2"])
    return result

