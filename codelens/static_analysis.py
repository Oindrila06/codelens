"""Run static analyzers (Semgrep when available, ruff security/bug rules as a fallback)."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from .schemas import Finding

log = logging.getLogger(__name__)

SEMGREP_SEVERITY = {"ERROR": "high", "WARNING": "medium", "INFO": "low"}
# ruff rule families: S = bandit security, B = bugbear, F = pyflakes (real bugs, not style)
RUFF_SELECT = "S,B,F"
RUFF_SEVERITY_PREFIX = {"S": "high", "B": "medium", "F": "medium"}


def _run(cmd: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def run_semgrep(repo_dir: Path, paths: list[str], config: str) -> list[Finding] | None:
    """Returns None when semgrep is unavailable or failed (so the caller can fall back)."""
    if not shutil.which("semgrep") or not paths:
        return None
    try:
        proc = _run(["semgrep", "--config", config, "--json", "--quiet", "--metrics=off", *paths], repo_dir)
        data = json.loads(proc.stdout or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        log.warning("semgrep failed: %s", exc)
        return None
    findings: list[Finding] = []
    for i, r in enumerate(data.get("results", [])):
        extra = r.get("extra", {})
        findings.append(
            Finding(
                id=f"sg{i}",
                path=str(r["path"]).replace("\\", "/"),
                line=r["start"]["line"],
                end_line=r["end"]["line"],
                rule=r.get("check_id", "semgrep"),
                message=(extra.get("message") or "").strip(),
                severity=SEMGREP_SEVERITY.get(str(extra.get("severity", "")).upper(), "medium"),
                source="static",
            )
        )
    return findings


def run_ruff(repo_dir: Path, paths: list[str]) -> list[Finding]:
    if not shutil.which("ruff") or not paths:
        return []
    try:
        proc = _run(
            ["ruff", "check", "--select", RUFF_SELECT, "--output-format", "json", "--no-cache",
             "--isolated", *paths],
            repo_dir,
        )
        data = json.loads(proc.stdout or "[]")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        log.warning("ruff failed: %s", exc)
        return []
    findings: list[Finding] = []
    for i, r in enumerate(data):
        code = r.get("code") or ""
        try:
            rel = Path(r["filename"]).resolve().relative_to(repo_dir.resolve()).as_posix()
        except ValueError:
            rel = Path(r["filename"]).as_posix()
        findings.append(
            Finding(
                id=f"rf{i}",
                path=rel,
                line=r["location"]["row"],
                end_line=r.get("end_location", r["location"])["row"],
                rule=code,
                message=r.get("message", ""),
                severity=RUFF_SEVERITY_PREFIX.get(code[:1], "low"),
                source="static",
            )
        )
    return findings


def analyze(repo_dir: Path, paths: list[str], semgrep_config: str) -> tuple[list[Finding], str]:
    """Run the best available analyzer. Returns (findings, tool_name)."""
    found = run_semgrep(repo_dir, paths, semgrep_config)
    if found is not None:
        return found, "semgrep"
    return run_ruff(repo_dir, paths), "ruff"


def only_on_lines(findings: list[Finding], added_by_path: dict[str, set[int]]) -> list[Finding]:
    """Keep findings that touch lines added by this PR; pre-existing issues are not the author's."""
    out = []
    for f in findings:
        added = added_by_path.get(f.path, set())
        if any(n in added for n in range(f.line, f.end_line + 1)):
            out.append(f)
    return out
