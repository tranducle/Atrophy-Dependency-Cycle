#!/usr/bin/env python3
"""Create a reproducibility manifest with relative artifact paths."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "artifact_manifest.json"

ARTIFACTS = [
    "README.md",
    "requirements.txt",
    "generate_figures.py",
    "abm_simulation_canonical.py",
    "robustness_ablation_canonical.py",
    "make_artifact_manifest.py",
    "simulation_results_canonical.json",
    "robustness_ablation_canonical.json",
    "sensitivity_analysis_canonical.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(rel_path: str) -> dict[str, object]:
    path = ROOT / rel_path
    if not path.exists():
        return {"path": rel_path, "exists": False}
    stat = path.stat()
    return {
        "path": rel_path,
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def main() -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "reproduction_order": [
            "abm_simulation_canonical.py",
            "robustness_ablation_canonical.py",
            "generate_figures.py",
            "make_artifact_manifest.py",
        ],
        "artifacts": [describe(rel_path) for rel_path in ARTIFACTS],
    }
    OUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote artifact_manifest.json")


if __name__ == "__main__":
    main()
