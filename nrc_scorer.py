#!/usr/bin/env python3
"""NRC word-list "primary emotion" scorer (offline, no model calls).

Scores each review's words against the public NRC Emotion Lexicon
(nrclex nrc_en.json: word -> {anger, anticipation, disgust, fear, joy,
sadness, surprise, trust, positive, negative}). The 8 standard emotions are
scored; the primary emotion is the emotion with the highest summed score.

Faithful to the request: tokenize on word boundaries, lowercase, look each
token up, add +1 to every emotion the word is associated with, sum per
emotion, take the argmax. No negation/valence model.

Tie-breaking: on ties, the canonically first emotion in this order wins
(anger, anticipation, disgust, fear, joy, sadness, surprise, trust);
ties are also reported so the ambiguity is visible.

Usage:
  nrc_scorer.py <review.jsonl> --out <augmented.jsonl>

Reads records with `text` (+ `title`) and a `label`/`review_id`; appends
`nrc_emotion`, `nrc_scores`, `nrc_tied`, `nrc_hits` and writes them out.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import nrclex

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


def load_lexicon():
    path = os.path.join(os.path.dirname(nrclex.__file__), "data", "nrc_en.json")
    raw = json.load(open(path, encoding="utf-8"))
    # keep only the 8 emotion labels per word
    lex = {}
    for word, labels in raw.items():
        emo = [l for l in labels if l in EMOTIONS]
        if emo:
            lex[word] = emo
    return lex


def derive(lex, title: str, text: str):
    """Return (primary_emotion|None, scores:dict, hits:int, tied:bool)."""
    scores = {e: 0 for e in EMOTIONS}
    hits = 0
    blob = f"{title or ''} {text or ''}"
    for tok in WORD_RE.findall(blob.lower()):
        emo = lex.get(tok)
        if emo:
            hits += 1
            for e in emo:
                scores[e] += 1
    best = max(scores.values())
    if best == 0:
        return None, scores, hits, False
    winners = [e for e in EMOTIONS if scores[e] == best]
    return winners[0], scores, hits, len(winners) > 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lex = load_lexicon()
    n_all = n_emotion = n_tie = 0
    n_hits_total = 0
    out = []
    for line in open(args.input, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not rec.get("label"):
            # keep dropped rows untouched (no emotion yet)
            out.append(rec)
            continue
        n_all += 1
        emo, scores, hits, tied = derive(lex, rec.get("title") or "", rec.get("text") or "")
        rec["nrc_emotion"] = emo
        rec["nrc_scores"] = scores
        rec["nrc_hits"] = hits
        rec["nrc_tied"] = tied
        out.append(rec)
        if emo:
            n_emotion += 1
        if tied:
            n_tie += 1
        n_hits_total += hits

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    print(f"lexicon entries (8-emotion labels): {len(lex)}")
    print(f"labeled reviews              : {n_all}")
    print(f"reviews with an NRC emotion   : {n_emotion}  ({100*n_emotion//max(n_all,1)}%)")
    print(f"reviews with NO matching word : {n_all - n_emotion}")
    print(f"reviews with a tied argmax    : {n_tie}")
    print(f"total lexicon word-hits      : {n_hits_total}  (avg {n_hits_total/max(n_all,1):.1f}/review)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
