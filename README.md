# CodeLens — context-aware AI code review for GitHub pull requests

CodeLens is a GitHub App that reviews pull requests the way a careful teammate would: it reads
the diff **and the code around it** (callers, callees, tests, repo conventions), combines LLM
reasoning with static analysis, **verifies every comment before posting**, and returns concrete
one-click fixes instead of generic advice.

**The problem:** diff-only AI reviewers are noisy and blind to the rest of the repo. They miss
the bug where a function's signature changes and a caller in another file breaks, and they bury
real issues under vague suggestions ("consider adding error handling").

**Approach**

| Idea | How it is implemented |
|---|---|
| Understand context across files | tree-sitter chunks every function/class, builds a call/import graph, hybrid BM25 + embedding retrieval, graph expansion (callers, callees, tests) into a token budget |
| Catch cross-file breakage deterministically | `graph.py` diffs function signatures between base and head and lists call sites outside the PR |
| Ground LLM output in facts | Semgrep (or ruff security rules) findings are *triaged* by the LLM instead of asking it to hunt for vulnerabilities cold |
| Avoid generic comments | Each comment must quote evidence that exists in the code, mention concrete identifiers, and anchor to a line in the diff; otherwise it is dropped |
| Suggestions that don't break code | Suggested patches are applied in memory, must parse, and must not add new pyflakes errors |
| Second opinion | Optional LLM verification pass discards comments the verifier judges invalid |
| Measurable | `evaluation/` compares diff-only LLM vs static-only vs full pipeline vs ablations on labeled PRs |
| Learns from use | 👍/👎 reactions and resolved threads are stored, reported, and can train a small LoRA triage filter |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the diagram and design decisions.

---

## 1. Quickstart (local, about 20 minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
cp .env.example .env            # then fill it in (steps below)
pytest -q                       # 67 tests, no network or API key needed
```

### Create the GitHub App
GitHub → Settings → Developer settings → GitHub Apps → **New GitHub App**

* **Webhook URL:** your smee/ngrok URL + `/webhook` (see below). **Webhook secret:** any long random string.
* **Repository permissions:** Pull requests = *Read & write*, Contents = *Read-only*, Metadata = *Read-only*.
* **Subscribe to events:** *Pull request*, *Pull request review thread*.
* After creating: note the **App ID**, click **Generate a private key**, save it as `secrets/codelens.private-key.pem`.
* **Install App** on a test repository (your own or your lab's).

Put these in `.env`: `GITHUB_APP_ID`, `GITHUB_WEBHOOK_SECRET`, `GITHUB_PRIVATE_KEY_PATH`, `ANTHROPIC_API_KEY`.
Optionally set `LLM_PRICE_IN_PER_MTOK` / `LLM_PRICE_OUT_PER_MTOK` so cost numbers are reported.

### Run it
```bash
# terminal 1: webhook tunnel (go to https://smee.io, create a channel, use its URL as the App's webhook URL)
npx smee-client --url https://smee.io/YOUR_CHANNEL --target http://localhost:8000/webhook

# terminal 2: the app (QUEUE_MODE=inline runs reviews in-process, no Redis needed)
uvicorn codelens.server:app --reload --port 8000
```
Open a pull request containing Python changes on the test repo. A review should appear within a
minute or so. Check `GET /healthz`, the terminal logs, and GitHub's App → *Advanced* → *Recent
Deliveries* (you can redeliver events from there; duplicates are ignored by design).

**Optional upgrades:** `pip install -r requirements-optional.txt` gives you Semgrep (richer rules than
the ruff fallback) and sentence-transformers (set `EMBEDDINGS_BACKEND=sentence-transformers`).

## 2. Deploy

```bash
docker compose up --build       # API + RQ worker + Redis; data persisted in ./data
```
Host it on Render, Railway, Fly.io, or a small VPS; expose port 8000 and point the App's webhook URL
at `https://YOUR_HOST/webhook`. Use `QUEUE_MODE=redis` in production so a slow review never blocks
webhook responses and failed jobs are retried (see `codelens/jobs.py`).

## 3. Per-repository configuration

Add `.codelens.yml` to a repo (see `.codelens.yml.example`): severity and confidence thresholds, max
comments per review, ignored paths, per-category on/off switches, `llm_verify`, and free-text team
conventions the reviewer should respect.

## 4. How a review runs

1. `server.py` verifies the HMAC signature, deduplicates by delivery id, enqueues, returns 200.
2. `jobs.py` fetches the PR, shallow-fetches base+head commits, loads `.codelens.yml`.
3. `indexer/` incrementally re-indexes only changed files (content hashes).
4. `static_analysis.py` runs Semgrep/ruff and keeps findings on **added lines only**.
5. `graph.py` flags changed/removed function signatures that still have callers outside the PR.
6. `pipeline.py` builds, per changed file, the changed code units plus retrieved context within a token budget.
7. `reviewer.py` asks the LLM for structured JSON comments and to triage the findings.
8. `verifier.py` drops comments that are unanchored, ungrounded, generic, below threshold, or whose
   suggested patch does not parse; an optional LLM pass drops the rest.
9. Comments are posted as one review (falling back to individual comments if GitHub rejects an anchor).
10. Posted comment ids are stored so reactions and resolved threads become feedback data.

## 5. Evaluation (do this before making any claims)

```bash
python -m evaluation.run_eval --llm none --configs static_only                  # no API key
python -m evaluation.run_eval --configs diff_only,static_only,full,full_no_verify --out evaluation/results/run1.json
python -m evaluation.run_eval --configs full,no_context,no_static,no_graph       # ablations
```
Outputs precision, recall, F1, comments/PR, **retrieval recall** (did the context include the files
needed to understand the bug?), tokens, latency and cost per PR.

* The 5 bundled cases in `evaluation/cases/` are **tiny smoke tests** showing the format. They are
  too easy to be evidence. Build **30-50 real cases**.
* Fast way to get real labeled cases: take bug/security **fix commits** from open-source Python repos
  (CVE-fix commits, or the CVEfixes dataset) and run
  `python -m evaluation.make_case_from_fix --repo PATH --fix SHA --id NAME --out evaluation/cases/NAME.json`.
  It reviews the *reverse* of the fix, so the lines the fix removed become the expected findings.
  Fill in `needed_context` by hand where a cross-file understanding is required.
* Add 10+ clean PRs (`"expected": []`) to measure false positives.
* Report the table, then write a short failure analysis of what each configuration got wrong.

## 6. Learning from real usage

```bash
python scripts/collect_feedback.py     # poll 👍/👎 reactions (run on a schedule; reactions send no webhooks)
python scripts/feedback_report.py      # reviews, comments, cost, acceptance rate, feedback coverage
```
Install it on your lab's repos for a few weeks, ask 5-10 developers for feedback, and report the
acceptance rate **together with how many comments received a signal**.

## 7. Fine-tuning (optional, only with enough data)

```bash
python -m finetune.prepare_data --out finetune/data
pip install torch transformers peft accelerate      # on Colab/Kaggle GPU is fine
python -m finetune.train_lora --train finetune/data/train.jsonl --val finetune/data/val.jsonl --out models/triage-lora
export TRIAGE_MODEL_DIR=models/triage-lora          # now used as a filter in verifier.py
```
Trains a small LoRA classifier that predicts whether a comment will be found useful. Data is split
**by repository** to avoid leakage. You need a few hundred labeled comments; otherwise skip this and rely
on the verification pass. Always compare against the prompted-LLM baseline on accuracy, latency and
cost, and report the result honestly even if the fine-tuned model does not win.

## 8. Project layout

```
codelens/            server, jobs, pipeline, verifier, reviewer, prompts, github_api, storage, graph, ...
codelens/indexer/    tree-sitter parser, embeddings, incremental SQLite index, hybrid retrieval, context budgeting
evaluation/          labeled-case runner, metrics, case builders
finetune/            triage dataset builder + LoRA training
scripts/             feedback collection and usage report
tests/               67 tests (webhooks, auth, diffs, parsing, retrieval, verifier, full pipeline, jobs)
```

## 9. Known limitations (state these in interviews)

* **Python only.** The parser and graph are language-specific; a second language is an extension.
* **Name-based call graph.** Python is dynamic, so callers are matched by function name plus import
  evidence, not full type resolution. Expect false positives for common names (some are filtered) and
  misses for dynamically dispatched calls.
* **Suggestions needing a new import** are dropped by design (a one-line patch cannot add the import).
* **Token counts are estimated** (~4 chars/token) for budgeting; real usage is reported from the API.
* **The index lives in memory at query time** (NumPy + inverted index); fine for repos up to tens of
  thousands of functions, but a vector database would be needed beyond that.
* **Fork PRs** need the fetch-by-ref fallback to succeed; private submodules are not handled.
* The hashing embedder is a weaker semantic model than sentence-transformers; compare them in your ablations.
* Not yet implemented: language-specific rules for C/embedded, mining past human review comments
  for conventions, check-run status reporting, running the repo's test suite to validate patches.

## 10. What has and has not been verified

Verified by the automated test-suite: webhook signature verification and deduplication, GitHub App JWT
(RS256) generation, token caching and pagination (mocked HTTP), diff parsing, tree-sitter parsing,
incremental indexing, retrieval, signature-change detection, comment verification, the whole
pipeline on real git repositories with a scripted LLM, and the end-to-end job (fetch, review, post,
record) against a fake GitHub client.

**Not verified here:** live calls to GitHub and the Anthropic API, the Redis/RQ queue path and the Docker
build (no Redis or Docker in the build environment), Semgrep with the registry rules,
sentence-transformers embeddings, and the LoRA training script (it needs a GPU). Expect to spend a
little time on a first live run (permissions, model name via `LLM_MODEL`, tunnel URL). Everything
else is designed so those failures are loud and isolated (one bad file or API error never kills a review).

## Resume bullets (fill in YOUR measured numbers)

> Built **CodeLens**, a GitHub App that reviews pull requests using code-graph retrieval (tree-sitter),
> static-analysis triage and LLM verification; improved issue recall from **X%** to **Y%** versus diff-only
> LLM review on **N** labeled vulnerable commits at **Z%** fewer comments per PR.
> Deployed on **N** repositories (**N** PRs, **Z%** suggestion acceptance); webhook pipeline with HMAC
> verification, idempotent queued jobs and incremental indexing. Stack: Python, FastAPI, Redis/RQ, SQLite, Semgrep, Docker.
