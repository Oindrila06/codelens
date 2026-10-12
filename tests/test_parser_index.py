from codelens.indexer.embeddings import HashingEmbedder, tokenize
from codelens.indexer.index import RepoIndex
from codelens.indexer.parser import parse_python, signatures
from codelens.indexer.retrieval import BM25, Retriever

SRC = '''import os
from a.b import c as d

X = setup()

class A(Base):
    attr = make()

    def m(self, x: int) -> int:
        return foo(x) + self.bar(1)

@decorator
def f(a, b=1):
    return helper(a)
'''


def test_chunks_and_signatures():
    p = parse_python("pkg/mod.py", SRC)
    names = {c.qualname: c for c in p.chunks}
    assert set(names) == {"A", "A.m", "f", "<module>"}
    assert names["f"].start_line == 12  # decorator included
    assert signatures(p)["A.m"] == "m(self, x: int) -> int"


def test_refs_attributed_to_enclosing_chunk():
    p = parse_python("pkg/mod.py", SRC)
    calls = {(r.name, p.chunks[r.chunk_index].qualname) for r in p.refs if r.kind == "call"}
    assert ("foo", "A.m") in calls and ("bar", "A.m") in calls
    assert ("helper", "f") in calls and ("setup", "<module>") in calls
    imports = {(r.name, r.module) for r in p.refs if r.kind == "import"}
    assert ("c", "a.b") in imports and ("os", "os") in imports


def test_test_files_flagged():
    assert parse_python("tests/test_x.py", "def test_a():\n    pass\n").chunks[0].is_test
    assert not parse_python("src/x.py", "def a():\n    pass\n").chunks[0].is_test


def test_syntax_error_does_not_crash():
    parse_python("bad.py", "def (:\n")


def test_tokenizer_splits_identifiers():
    toks = tokenize("getUserById snake_case_name")
    assert {"get", "user", "by", "id", "snake", "case", "name"}.issubset(toks)


def test_hashing_embedder_similarity():
    e = HashingEmbedder()
    a, b, c = e.embed(["def get_user(uid): return db.query_user(uid)",
                       "def fetch_user(uid): return db.query_user(uid)",
                       "def render_chart(ax): ax.plot(x, y)"])
    assert float(a @ b) > float(a @ c)


def test_bm25_ranks_matching_doc_first():
    bm = BM25([["alpha", "beta"], ["gamma", "delta"], ["alpha", "alpha", "zeta"]])
    scores = bm.scores(["alpha"])
    assert max(scores, key=scores.get) == 2 and 1 not in scores


def _repo(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "utils.py").write_text("def get_user(uid):\n    return db.query(uid)\n")
    (tmp_path / "api.py").write_text("from utils import get_user\n\ndef handler(req):\n    return get_user(req.id)\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_api.py").write_text("from utils import get_user\n\ndef test_get():\n    assert get_user(1)\n")
    return tmp_path


def test_incremental_indexing(tmp_path):
    repo = _repo(tmp_path / "r")
    idx = RepoIndex(tmp_path / "i.db", HashingEmbedder())
    assert idx.update(repo)["indexed"] == 3
    assert idx.update(repo) == {"indexed": 0, "unchanged": 3, "removed": 0}
    (repo / "api.py").write_text("def other():\n    pass\n")
    (repo / "tests" / "test_api.py").unlink()
    stats = idx.update(repo)
    assert stats["indexed"] == 1 and stats["removed"] == 1


def test_expand_finds_callers_and_tests(tmp_path):
    repo = _repo(tmp_path / "r")
    idx = RepoIndex(tmp_path / "i.db", HashingEmbedder())
    idx.update(repo)
    changed = [c for c in idx.chunks_in_path("utils.py") if c.name == "get_user"]
    items = Retriever(idx).expand(changed)
    cats = {(i.category, i.path) for i in items}
    assert ("caller", "api.py") in cats and ("test", "tests/test_api.py") in cats


def test_search_finds_relevant_chunk(tmp_path):
    repo = _repo(tmp_path / "r")
    idx = RepoIndex(tmp_path / "i.db", HashingEmbedder())
    idx.update(repo)
    hits = Retriever(idx).search("handler request", k=1)
    assert idx.get_chunk(hits[0][0]).name == "handler"
