"""Summarise real-world usage: reviews, comments, cost, and acceptance rate.

These are the numbers to put on your resume (once they are real).
    python scripts/feedback_report.py
"""
from __future__ import annotations

from codelens.config import get_settings
from codelens.storage import Storage


def main() -> None:
    st = Storage(get_settings().db_path)
    s = st.summary()
    fb = s["feedback"]
    accepted = fb.get("thumbs_up", 0) + fb.get("resolved", 0)
    rejected = fb.get("thumbs_down", 0)
    print(f"Reviews completed:      {s['reviews']}")
    print(f"Comments posted:        {s['comments_posted']}  (tracked: {s['tracked_comments']})")
    print(f"Avg review time:        {s['avg_seconds']:.1f}s")
    print(f"Tokens in/out:          {s['tokens_in']} / {s['tokens_out']}")
    print(f"Estimated cost:         ${s['cost']:.3f}")
    print(f"Feedback signals:       {fb or 'none yet'}")
    if accepted + rejected:
        print(f"Acceptance rate:        {accepted / (accepted + rejected):.0%} "
              f"({accepted} useful vs {rejected} not useful, of comments that received feedback)")
    if s["tracked_comments"]:
        print(f"Feedback coverage:      {(accepted + rejected) / s['tracked_comments']:.0%} of comments got a signal "
              "(report this next to the acceptance rate; a high rate on few signals is weak evidence)")


if __name__ == "__main__":
    main()
