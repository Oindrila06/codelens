from codelens.diffs import analyze_patch
from codelens.schemas import RepoConfig, ReviewComment
from codelens.verifier import (
    Verifier,
    apply_suggestion,
    clean_evidence,
    evidence_grounded,
    is_generic,
    suggestion_is_safe,
)

FILE = """import subprocess


def run(user_cmd):
    result = subprocess.run(user_cmd, shell=True)
    return result
"""
PATCH = """@@ -1,3 +1,6 @@
 import subprocess
+
+
+def run(user_cmd):
+    result = subprocess.run(user_cmd, shell=True)
+    return result"""


def mk(**kw):
    base = dict(path="a.py", line=5, severity="high", category="security", title="Shell injection",
                explanation="`user_cmd` is passed to subprocess.run with shell=True, so crafted input runs arbitrary commands.",
                evidence="result = subprocess.run(user_cmd, shell=True)", confidence=0.9)
    base.update(kw)
    return ReviewComment(**base)


def verifier(cfg=None, mode="full"):
    return Verifier({"a.py": FILE}, {"a.py": PATCH}, {"a.py": analyze_patch(PATCH)}, cfg or RepoConfig(), mode=mode)


def test_good_comment_kept():
    v = verifier()
    assert len(v.check([mk()])) == 1 and not v.dropped


def test_hallucinated_evidence_dropped():
    v = verifier()
    assert v.check([mk(evidence="os.system(cmd)")]) == []
    assert "evidence" in v.dropped[0][1]


def test_evidence_with_line_numbers_still_matches():
    assert evidence_grounded("    5 + result = subprocess.run(user_cmd, shell=True)", FILE, PATCH)
    assert clean_evidence("+x = 1\n+y = 2") == ["x = 1", "y = 2"]


def test_generic_comment_dropped():
    vague = dict(title="Code quality", explanation="Consider improving this code for better quality overall in future.")
    assert is_generic(mk(**vague))
    assert not is_generic(mk())
    assert is_generic(mk(explanation="Too short."))
    v = verifier()
    assert v.check([mk(**vague)]) == [] and v.dropped[0][1] == "generic comment"


def test_line_snapped_or_dropped():
    v = verifier()
    kept = v.check([mk(line=8)])  # 8 is outside the diff, but within 3 lines of line 6
    assert kept and kept[0].line == 6
    v2 = verifier()
    assert v2.check([mk(line=40)]) == [] and v2.dropped[0][1] == "line not in diff"


def test_unknown_file_dropped():
    assert verifier().check([mk(path="other.py")]) == []


def test_thresholds_and_disabled_categories():
    assert verifier(RepoConfig(min_confidence=0.95)).check([mk()]) == []
    assert verifier(RepoConfig(min_severity="critical")).check([mk()]) == []
    assert verifier(RepoConfig(checks={"security": False})).check([mk()]) == []


def test_invalid_suggestion_removed_but_comment_kept():
    c = mk(suggestion="result = subprocess.run(user_cmd, shell=True")  # unbalanced paren
    kept = verifier().check([c])
    assert len(kept) == 1 and kept[0].suggestion is None


def test_valid_suggestion_kept():
    c = mk(suggestion="    result = subprocess.run(user_cmd.split())")
    kept = verifier().check([c])
    assert kept[0].suggestion is not None


def test_suggestion_needing_new_import_is_dropped():
    # shlex is never imported, so applying this would introduce a NameError; the comment survives.
    c = mk(suggestion="    result = subprocess.run(shlex.split(user_cmd))")
    kept = verifier().check([c])
    assert len(kept) == 1 and kept[0].suggestion is None


def test_suggestion_helpers():
    lines = FILE.splitlines()
    assert "x = 1" in apply_suggestion(lines, 5, 5, "    x = 1")
    assert not suggestion_is_safe("a.py", FILE, 5, 5, "    result = subprocess.run(user_cmd, shell=True)")  # no-op
    assert not suggestion_is_safe("a.py", FILE, 5, 99, "x = 1")  # range outside file


def test_dedupe_and_cap():
    cfg = RepoConfig(max_comments=1)
    v = verifier(cfg)
    a, b = mk(severity="medium"), mk(severity="critical", title="Other")
    kept = v.check([a, b])
    assert len(kept) == 1 and kept[0].severity == "critical"


def test_basic_mode_skips_content_checks():
    v = verifier(mode="basic")
    assert len(v.check([mk(evidence="made up", confidence=0.1)])) == 1
