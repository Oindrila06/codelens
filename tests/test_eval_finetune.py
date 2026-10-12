import json
from pathlib import Path

from codelens.schemas import ReviewComment
from codelens.testing import build_repo
from evaluation.make_case_from_fix import build_case
from evaluation.materialize import load_cases, materialize
from evaluation.metrics import aggregate, retrieval_recall, score_case
from finetune.prepare_data import build_examples, split_by_repo


def rc(path, line, start=None):
    return ReviewComment(path=path, line=line, start_line=start)


def test_scoring_matches_with_tolerance_one_to_one():
    exp = [{"path": "a.py", "start_line": 10, "end_line": 12}, {"path": "b.py", "start_line": 1, "end_line": 1}]
    s = score_case([rc("a.py", 13), rc("a.py", 11), rc("c.py", 5)], exp)
    assert (s.tp, s.fp, s.fn) == (1, 2, 1)  # second a.py comment cannot double-match
    assert score_case([], []).tp == 0 and score_case([rc("a.py", 1)], []).fp == 1


def test_aggregate_and_retrieval_recall():
    a = aggregate([score_case([rc("a.py", 1)], [{"path": "a.py", "start_line": 1, "end_line": 1}]),
                   score_case([rc("x.py", 1)], [])])
    assert a["precision"] == 0.5 and a["recall"] == 1.0 and a["comments_per_pr"] == 1.0
    assert retrieval_recall({"a.py"}, ["a.py", "b.py"]) == 0.5 and retrieval_recall(set(), []) is None


def test_bundled_cases_load_and_materialize(tmp_path):
    cases = load_cases(Path("evaluation/cases"))
    assert len(cases) >= 5
    for case in cases:
        inp = materialize(case, tmp_path / case["id"])
        assert inp.files, case["id"]


def test_make_case_from_fix_labels_the_reverted_lines(tmp_path):
    fixed = {"a.py": "def f(x):\n    return safe(x)\n"}
    vulnerable = {"a.py": "def f(x):\n    return eval(x)\n"}
    repo, vuln_sha, fix_sha = build_repo(tmp_path / "r", vulnerable, fixed)
    case = build_case(repo, fix_sha, "c1", "security", "t")
    assert case["base"] == fix_sha and case["head"] == vuln_sha
    assert case["expected"] == [{"path": "a.py", "start_line": 2, "end_line": 2, "category": "security"}]
    inp = materialize(case, tmp_path / "w")
    assert inp.files[0].path == "a.py" and "eval(x)" in (inp.files[0].patch or "")
    json.dumps(case)


def test_finetune_split_is_by_repo_and_labels_are_clean():
    rows = []
    for i in range(30):
        sigs = {0: ["thumbs_up"], 1: ["thumbs_down"], 2: ["thumbs_up", "thumbs_down"], 3: []}[i % 4]
        rows.append({"gh_comment_id": i, "repo": f"o/r{i % 5}", "signals": sigs,
                     "payload": {"category": "bug", "severity": "high", "source": "llm", "title": "t",
                                 "explanation": "e", "evidence": "x"}})
    ex = build_examples(rows)
    assert all(e["label"] in (0, 1) for e in ex) and len(ex) < 30  # conflicting/no-signal rows dropped
    train, val = split_by_repo(ex)
    assert {e["repo"] for e in train}.isdisjoint({e["repo"] for e in val})
