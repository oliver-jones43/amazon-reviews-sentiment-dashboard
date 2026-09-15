#!/usr/bin/env python3
"""Verify the dashboard's embedded data + filtering against the source jsonl.

Checks (three-class rule: rating>=4 positive / ==3 neutral / <=2 negative):
  1. dashboard rows == source labeled rows as a multiset (by review_id),
  2. per row: embedded gold is self-consistent with its own rating and matches
     the source row's label,
  3. every filter count produced by the (ported) matchFilter equals the
     recorded ground-truth count,
  4. the row template emits a data-r review-id hook.

Usage: verify_dashboard.py <dashboard.html> <source.jsonl>
Exit 0 = all checks PASS.
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def gold_from_rating(r):
    if r is None:
        return None
    if r >= 4:
        return "positive"
    if r == 3:
        return "neutral"
    return "negative"


# matchFilter ported verbatim from the dashboard JS (operates on rows that carry `gold`).
def match_filter(r, f):
    if f == "ok":
        return r["label"] == r["gold"]
    if f == "bad":
        return r["label"] != r["gold"]
    if f == "pos":
        return r["gold"] == "positive"
    if f == "neu":
        return r["gold"] == "neutral"
    if f == "neg":
        return r["gold"] == "negative"
    return True  # all


def main(html_path, src_path):
    html = open(html_path, encoding="utf-8").read()
    dash = json.loads(html.split("const DASH=")[1].split("</script>")[0].strip().rstrip(";"))
    src = [r for r in (json.loads(l) for l in open(src_path, encoding="utf-8") if l.strip())
           if r.get("label")]

    fails = []
    dc = Counter(r["review_id"] for r in dash["rows"])
    sc = Counter(r["review_id"] for r in src)
    if dc != sc:
        fails.append(f"rows differ: dashboard {sum(dc.values())} vs source {sum(sc.values())}")

    src_by_id = {}
    for r in src:
        src_by_id.setdefault(r["review_id"], r)
    for r in dash["rows"]:
        g = gold_from_rating(r["rating"])
        if r["gold"] != g:
            fails.append(f"{r['review_id'][:30]}: embedded gold {r['gold']} != rating-derived {g}")
        s = src_by_id.get(r["review_id"])
        if s is None:
            fails.append(f"{r['review_id'][:30]}: no source row")
        elif r["label"] != s["label"]:
            fails.append(f"{r['review_id'][:30]}: label {r['label']} != source {s['label']}")

    filters = ["all", "ok", "bad", "pos", "neu", "neg"]
    rec = {
        "all": len(src),
        "ok": sum(1 for s in src if s["label"] == gold_from_rating(s["rating"])),
        "bad": sum(1 for s in src if s["label"] != gold_from_rating(s["rating"])),
        "pos": sum(1 for s in src if gold_from_rating(s["rating"]) == "positive"),
        "neu": sum(1 for s in src if gold_from_rating(s["rating"]) == "neutral"),
        "neg": sum(1 for s in src if gold_from_rating(s["rating"]) == "negative"),
    }
    got = {f: sum(1 for r in dash["rows"] if match_filter(r, f)) for f in filters}
    for f in filters:
        if rec[f] != got[f]:
            fails.append(f"filter {f}: recorded {rec[f]} vs rendered {got[f]}")

    has_data_r = 'data-r="${r.review_id}"' in html
    if not has_data_r:
        fails.append("row template missing data-r hook")

    print("=" * 60)
    print(f"verifying {html_path}")
    print("=" * 60)
    print(f"dashboard scored rows : {sum(dc.values())}  (agreement {dash['stats']['agreement']})")
    for f in filters:
        print(f"  filter {f:5}: recorded={rec[f]:3} rendered={got[f]:3}  "
              f"{'OK' if rec[f] == got[f] else 'FAIL'}")
    print("row template data-r hook:", "OK" if has_data_r else "FAIL")
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails[:25]:
            print("  -", f)
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
