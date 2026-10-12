"""End-to-end pipeline test on a real git repo with a scripted fake LLM (no network)."""
import json

from codelens.diffs import analyze_patch
from codelens.gitutil import diff_files
from codelens.llm import FakeLLM
from codelens.pipeline import PipelineOptions, run_review
from codelens.schemas import RepoConfig, ReviewInput
from codelens.testing import build_repo

BEFORE = {
    "utils.py": "def get_user(uid):\n    return db.query(uid)\n",
    "api.py": "from utils import get_user\n\n\ndef handler(req):\n    return get_user(req.id)\n",
    "run.py": "def noop():\n    return 1\n",
}
AFTER = {
    **BEFORE,
    "utils.py": "def get_user(uid, include_deleted):\n    return db.query(uid, include_deleted)\n",
    "run.py": (
        "import subprocess\n\n\ndef noop():\n    return 1\n\n\n"
        "def execute(cmd):\n    return subprocess.run(cmd, shell=True)\n"
    ),
}


def scripted(system, user):
    if "strict verifier" in system:
        return {"verdicts": [{"index": i, "verdict": "valid", "reason": "ok"} for i in range(5)]}
    if "File under review: run.py" in user:
        return {"comments": [
            {"path": "run.py", "line": 9, "severity": "high", "category": "security",
             "title": "Shell injection", "confidence": 0.9,
             "explanation": "`cmd` reaches subprocess.run with shell=True, so attacker-controlled input executes arbitrary commands.",
             "evidence": "return subprocess.run(cmd, shell=True)",
             "suggestion": "    return subprocess.run(cmd.split())"},
            {"path": "run.py", "line": 9, "severity": "low", "category": "design", "title": "Hallucinated",
             "confidence": 0.9, "explanation": "`os.system` is used here which is discouraged for several reasons.",
             "evidence": "os.system(cmd)"},
        ], "dismissed_findings": []}
    if "File under review: utils.py" in user:
        assert "api.py" in user  # graph finding and caller context must reach the model
        return {"comments": [
            {"path": "utils.py", "line": 1, "severity": "high", "category": "bug",
             "title": "Breaking signature change", "confidence": 0.85, "finding_id": "gr1",
             "explanation": "`get_user` now requires `include_deleted`, but `handler` in api.py still calls it with one argument.",
             "evidence": "def get_user(uid, include_deleted):", "related_files": ["api.py"]},
        ]}
    return {"comments": []}


def make_input(tmp_path):
    repo, base, head = build_repo(tmp_path / "repo", BEFORE, AFTER)
    return ReviewInput(repo_dir=repo, base_sha=base, head_sha=head, files=diff_files(repo, base, head),
                       title="Refactor", config=RepoConfig())


def test_diff_files_matches_github_shape(tmp_path):
    inp = make_input(tmp_path)
    by = {f.path: f for f in inp.files}
    assert set(by) == {"utils.py", "run.py"}
    assert by["run.py"].patch.startswith("@@") and by["run.py"].additions > 0
    assert 9 in analyze_patch(by["run.py"].patch).added


def test_full_pipeline_finds_cross_file_and_security_issues(tmp_path):
    inp = make_input(tmp_path)
    llm = FakeLLM(handler=scripted)
    res = run_review(inp, llm, opts=PipelineOptions(use_static=False), index_db=tmp_path / "i.db")
    titles = {c.title for c in res.comments}
    assert titles == {"Shell injection", "Breaking signature change"}
    assert "api.py" in res.context_paths  # retrieval pulled in the caller
    assert any(f.rule == "signature-changed" for f in res.findings)
    assert any(why == "evidence not found in code" for _, why in res.dropped)
    shell = next(c for c in res.comments if c.title == "Shell injection")
    assert shell.suggestion and shell.source == "llm"
    assert res.usage.calls >= 2 and res.usage.input_tokens > 0


def test_diff_only_baseline_cannot_see_callers(tmp_path):
    inp = make_input(tmp_path)
    seen = []

    def handler(system, user):
        seen.append(user)
        return {"comments": []}

    opts = PipelineOptions(use_context=False, use_static=False, use_graph=False, verify="basic")
    res = run_review(inp, FakeLLM(handler=handler), opts=opts)
    assert res.comments == [] and res.context_paths == set()
    assert all("api.py" not in u.split("Static analysis")[0] for u in seen)


def test_static_only_baseline_uses_ruff_without_llm(tmp_path):
    inp = make_input(tmp_path)
    opts = PipelineOptions(use_context=False, use_graph=False, llm_review=False, verify="basic")
    res = run_review(inp, None, opts=opts)
    assert res.static_tool in ("semgrep", "ruff")
    assert any(c.path == "run.py" and c.source == "static" for c in res.comments)  # S602 shell=True


def test_ignored_and_non_python_files_skipped(tmp_path):
    repo, base, head = build_repo(tmp_path / "r", {"a.py": "x = 1\n"}, {"a.py": "x = 2\n", "README.md": "hi\n"})
    cfg = RepoConfig(ignore_paths=["a.py"])
    inp = ReviewInput(repo_dir=repo, base_sha=base, head_sha=head, files=diff_files(repo, base, head), config=cfg)
    res = run_review(inp, FakeLLM(handler=scripted), opts=PipelineOptions(use_static=False), index_db=tmp_path / "i.db")
    assert res.comments == [] and res.files_reviewed == 0
    assert any("ignored" in s for s in res.skipped) and any("not a Python" in s for s in res.skipped)


def test_llm_failure_on_one_file_does_not_crash(tmp_path):
    inp = make_input(tmp_path)

    def boom(system, user):
        raise RuntimeError("api down")

    res = run_review(inp, FakeLLM(handler=boom), opts=PipelineOptions(use_static=False), index_db=tmp_path / "i.db")
    assert res.comments == []


def test_findings_are_json_serialisable(tmp_path):
    inp = make_input(tmp_path)
    res = run_review(inp, FakeLLM(handler=scripted), opts=PipelineOptions(use_static=False), index_db=tmp_path / "i.db")
    json.dumps([c.model_dump() for c in res.comments])
