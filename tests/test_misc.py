import pytest

from codelens.llm import extract_json
from codelens.repo_config import is_ignored, load_repo_config
from codelens.schemas import ReviewComment


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Here you go:\n```json\n{"a": [1, 2]}\n```\nthanks') == {"a": [1, 2]}
    assert extract_json('noise {"comments": []} noise') == {"comments": []}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_comment_validators_are_lenient():
    c = ReviewComment(path="a.py", line="7", severity="SEVERE", category="weird", confidence="2.5", suggestion="  ")
    assert (c.line, c.severity, c.category, c.confidence, c.suggestion) == (7, "low", "other", 1.0, None)


def test_repo_config_loading(tmp_path):
    assert load_repo_config(tmp_path).max_comments == 8
    (tmp_path / ".codelens.yml").write_text("max_comments: 3\nignore_paths: ['docs/*']\nchecks: {design: false}\n")
    cfg = load_repo_config(tmp_path)
    assert cfg.max_comments == 3 and not cfg.category_enabled("design") and cfg.category_enabled("bug")
    (tmp_path / ".codelens.yml").write_text("max_comments: [not, an, int]\n")
    assert load_repo_config(tmp_path).max_comments == 8  # invalid config falls back to defaults
    assert is_ignored("docs/a/b.py", ["docs/*"]) and not is_ignored("src/a.py", ["docs/*"])
