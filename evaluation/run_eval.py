"""Run baseline vs. full-pipeline comparisons over labeled cases and print a results table.

    python -m evaluation.run_eval --cases evaluation/cases --configs diff_only,static_only,full
    python -m evaluation.run_eval --llm none --configs static_only      # no API key needed
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from codelens.config import get_settings
from codelens.llm import AnthropicLLM
from codelens.pipeline import PipelineOptions, run_review

from .materialize import load_cases, materialize
from .metrics import CaseScore, aggregate, retrieval_recall, score_case

CONFIGS: dict[str, PipelineOptions] = {
    # A: the common "send the diff to an LLM" approach
    "diff_only": PipelineOptions(use_context=False, use_static=False, use_graph=False, verify="basic"),
    # B: static analysis alone (no LLM)
    "static_only": PipelineOptions(use_context=False, use_static=True, use_graph=False, llm_review=False, verify="basic"),
    # C: everything on
    "full": PipelineOptions(),
    # Ablations of C
    "full_no_verify": PipelineOptions(verify="basic", llm_verify=False),
    "no_context": PipelineOptions(use_context=False),
    "no_static": PipelineOptions(use_static=False),
    "no_graph": PipelineOptions(use_graph=False),
}
NEEDS_LLM = {name for name, o in CONFIGS.items() if o.llm_review}


def fmt(x, pct=False, nd=2):
    if x is None:
        return "–"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="evaluation/cases")
    ap.add_argument("--configs", default="diff_only,static_only,full,full_no_verify")
    ap.add_argument("--llm", choices=["anthropic", "none"], default="anthropic")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    settings = get_settings()
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {unknown}. Choose from {sorted(CONFIGS)}")
    if args.llm == "none":
        names = [n for n in names if n not in NEEDS_LLM]
    llm = AnthropicLLM(settings) if args.llm == "anthropic" and any(n in NEEDS_LLM for n in names) else None

    cases = load_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} case(s), configs: {', '.join(names)}\n")

    rows, details = [], {}
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        for name in names:
            opts = CONFIGS[name]
            scores: list[CaseScore] = []
            seconds = tokens = 0.0
            cost = 0.0
            per_case = []
            for case in cases:
                inp = materialize(case, tmp / f"{name}-{case['id']}")
                t0 = time.time()
                res = run_review(inp, llm if opts.llm_review else None, settings, opts,
                                 index_db=tmp / f"{name}-{case['id']}.db")
                seconds += time.time() - t0
                tokens += res.usage.input_tokens + res.usage.output_tokens
                cost += res.usage.cost(settings.price_in_per_mtok, settings.price_out_per_mtok)
                s = score_case(res.comments, case.get("expected", []))
                if opts.use_context:
                    s.retrieval_recall = retrieval_recall(res.context_paths, case.get("needed_context", []))
                scores.append(s)
                per_case.append({
                    "case": case["id"], "tp": s.tp, "fp": s.fp, "fn": s.fn,
                    "comments": [{"path": c.path, "line": c.line, "title": c.title, "severity": c.severity}
                                 for c in res.comments],
                    "dropped": [{"title": c.title, "why": why} for c, why in res.dropped],
                })
            agg = aggregate(scores)
            n = max(len(cases), 1)
            agg.update(config=name, tokens_per_pr=tokens / n, seconds_per_pr=seconds / n, cost_per_pr=cost / n)
            rows.append(agg)
            details[name] = per_case

    header = "| config | precision | recall | F1 | comments/PR | retrieval recall | tokens/PR | s/PR | $/PR |"
    print(header + "\n" + "|" + "---|" * 9)
    for r in rows:
        print(f"| {r['config']} | {fmt(r['precision'], True)} | {fmt(r['recall'], True)} | {fmt(r['f1'])} | "
              f"{fmt(r['comments_per_pr'], nd=1)} | {fmt(r['retrieval_recall'], True)} | "
              f"{r['tokens_per_pr']:.0f} | {r['seconds_per_pr']:.1f} | {r['cost_per_pr']:.3f} |")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": rows, "details": details}, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
