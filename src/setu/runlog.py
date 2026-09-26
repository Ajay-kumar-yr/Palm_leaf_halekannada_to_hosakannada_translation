"""Shared run-folder bookkeeping (CLAUDE.md rule 4: every run writes
`runs/<timestamp>/` with its config, results, and git commit hash -- a
result without a run folder does not exist).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as e:  # no git / not a repo -- fail loudly, don't fake a hash
        raise RuntimeError(f"Could not determine git commit hash for run folder: {e}")


def start_run(name: str, config: dict[str, Any]) -> Path:
    """Create runs/<timestamp>_<name>/ and write config.json + git commit hash."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO_ROOT / "runs" / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(_git_commit_hash() + "\n", encoding="utf-8")
    return run_dir


def finish_run(run_dir: Path, results: dict[str, Any]) -> None:
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
