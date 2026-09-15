#!/usr/bin/env python3
"""Recompute every number published in README.md from the saved raw output.

This is your self-service way to re-check the report: run it on the balanced
run's raw output and compare the printed values with the README tables. No
model calls, no dashboard required — the figures are derived directly from
`reviews_3class.jsonl` (label, rating, text, title, emotion, review_id).

Usage:  python recompute_run.py <reviews_3class.jsonl>
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from nrc_scorer import EMOTIONS, derive, load_lexicon

VALENCE_POS = {"joy", "trust", "anticipation", "surprise"}
VALENCE_NEG = {"anger", "disgust", "fear", "sadness"}


def gold(rating):
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    lab = [r for r in rows if r.get("label")]
    lex = load_lexicon()

    n = len(lab)
    agree = sum(1 for r in lab if r["label"] == gold(r["rating"]))
    print("=" * 56)
    print("PUBLISHED NUMBERS — recomputed from the RAW OUTPUT")
    print(f"source: {path}")
    print("=" * 56)
    print(f"scored           : {n}   (of {len(rows)} submitted)")
    print(f"3-class AGREEMENT: {agree}/{n} = {100*agree/n:.1f}%")
    print("-" * 56)
    print("Per-class recall (gold=rating class, pred=model):")
    cls = ["positive", "neutral", "negative"]
    print(f"  {'class':9} {'correct':>8} {'n':>4} {'recall':>8}")
    conf = {g: {p: 0 for p in cls} for g in cls}
    for r in lab:
        conf[gold(r["rating"])][r["label"]] += 1
    for g in cls:
        tot = sum(conf[g].values())
        ok = conf[g][g]
        print(f"  {g:9} {ok:>7}/{tot:<3} {tot:>4} {100*ok/tot:>7.1f}%")
    print("-" * 56)
    print("Confusion (rows = gold / rating, cols = model):")
    gp = "gold\\pred"
    print(f"  {gp:11}" + "".join(f"{p:>9}" for p in cls))
    for g in cls:
        print(f"  {g:11}" + "".join(f"{conf[g][p]:>9}" for p in cls))
    print("-" * 56)
    star = Counter()
    for r in lab:
        star[int(round(r["rating"]))] += 1
    print("Star distribution: " + ", ".join(f"{s}\u2605={star[s]}" for s in range(1, 6)))
    print("-" * 56)
    # emotion comparison
    both, agree_e, valence, miss, ties = 0, 0, 0, 0, 0
    llm_d, nrc_d = Counter(), Counter()
    for r in lab:
        nrc, _, _, tied = derive(lex, r.get("title") or "", r.get("text") or "")
        if nrc is None:
            miss += 1
        if tied:
            ties += 1
        le = r.get("emotion")
        if le:
            llm_d[le] += 1
        if nrc:
            nrc_d[nrc] += 1
        if le and nrc:
            both += 1
            if le == nrc:
                agree_e += 1
            if (le in VALENCE_POS and nrc in VALENCE_POS) or (le in VALENCE_NEG and nrc in VALENCE_NEG):
                valence += 1
    print("Emotions (LLM vs NRC):")
    print(f"  both primary emotions : {both}")
    print(f"  same emotion          : {agree_e}/{both} = {100*agree_e/both:.1f}%")
    print(f"  same valence          : {valence}/{both} = {100*valence/both:.1f}%")
    print(f"  NRC no emotion word   : {miss} of {n}")
    print(f"  NRC tied argmax       : {ties} of {n}")
    print(f"  LLM dist              : " + ", ".join(f"{e}={llm_d.get(e,0)}" for e in EMOTIONS))
    print(f"  NRC  dist             : " + ", ".join(f"{e}={nrc_d.get(e,0)}" for e in EMOTIONS))
    print("=" * 56)
    print("Compare every line above with the README tables — they must match.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reviews_3class.jsonl")
