#!/usr/bin/env python3
"""
Binary review sentiment classifier using an OpenAI-compatible LLM endpoint.

Classifies Amazon product reviews as POSITIVE or NEGATIVE based on the review
title and free-text using a chat-completions LLM (OpenAI-compatible API).

Endpoint / model / key are resolved from (arg > env > default):
    --base-url       OPENAI_BASE_URL        default https://api.openai.com/v1
    --api-key        OPENAI_API_KEY         (required)
    --model          OPENAI_MODEL           default gpt-4o-mini

Features
--------
* Streaming reads from a .jsonl.gz input (no full decompression in RAM).
* Concurrent workers with retry + exponential backoff on 429/5xx.
* Structured JSON output from the model, validated + tenaciously parsed.
* Incremental checkpointing: processed ASINs are recorded so a re-run resumes.
* --limit/--sample to control cost.
* --eval compares LLM labels against a rating-based heuristic on a sample.

Output
------
* <output>.jsonl   one record per review: asin, sentiment, confidence
* <output>.summary  counts + (with --eval) confusion matrix vs rating heuristic
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

SENTIMENTS = ("positive", "neutral", "negative")
EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
EMO_RE = {e: re.compile(rf"\b{re.escape(e)}\b", re.IGNORECASE) for e in EMOTIONS}


def emotion_from_text(text: str) -> str | None:
    """Find any of the eight NRC emotions mentioned literally in the reply text."""
    for e in EMOTIONS:
        if EMO_RE[e].search(text or ""):
            return e
    return None


# --------------------------------------------------------------------------- #
#  Parsing helpers
# --------------------------------------------------------------------------- #
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOKEN_RE = re.compile(r"\b(positive|neutral|negative)\b", re.IGNORECASE)


def parse_model_output(raw: str) -> dict:
    """Tolerant parser for the model's reply.

    Accepts, in order of preference:
      1. A JSON object with {"sentiment": ..., "emotion": ..., "confidence": ...}
      2. A bare token "positive"/"negative" (+ an emotion word and optional confidence).
    Returns {"sentiment": str, "emotion": str|None, "confidence": float|None}
    on success, or all-None on failure.
    """
    if not raw:
        return {"sentiment": None, "emotion": None, "confidence": None}
    text = raw.strip()

    # 1) Try to find/parse a JSON object.
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "sentiment" in obj:
                sent = str(obj["sentiment"]).strip().lower()
                if sent in SENTIMENTS:
                    emo = str(obj.get("emotion", "")).strip().lower() or None
                    if emo not in EMOTIONS or emo == "":
                        emo = None
                    conf = obj.get("confidence")
                    try:
                        conf = float(conf)
                    except (TypeError, ValueError):
                        conf = None
                    conf = max(0.0, min(1.0, conf)) if conf is not None else None
                    return {"sentiment": sent, "emotion": emo, "confidence": conf}
        except json.JSONDecodeError:
            pass

    # 2) Bare token fallback.
    tok = _TOKEN_RE.search(text)
    if tok:
        sent = tok.group(1).lower()
        emo = emotion_from_text(text)
        conf = None
        after = re.sub(_TOKEN_RE, "", text).strip()
        if after:
            num = re.search(r"0\.\d+|\b\d+(?:\.\d+)?\b", after)
            if num:
                v = float(num.group(0))
                if 0.0 <= v <= 1.0:
                    conf = v
        return {"sentiment": sent, "emotion": emo, "confidence": conf}

    return {"sentiment": None, "emotion": None, "confidence": None}


# --------------------------------------------------------------------------- #
#  Model call
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a sentiment + emotion annotator for product reviews. Based only on the "
    "title and body text of each review: "
    "(1) classify the overall sentiment into EXACTLY one of THREE classes: "
    "POSITIVE (praising, satisfied, recommending), NEUTRAL (mixed, balanced, "
    "informational, or flatly factual — neither clearly positive nor negative), or "
    "NEGATIVE (critical, disappointed, complaining). "
    "(2) choose the single PRIMARY emotion expressed, from EXACTLY one of these eight: "
    "anger, anticipation, disgust, fear, joy, sadness, surprise, trust. "
    'Reply with ONLY valid JSON matching exactly: '
    '{"sentiment": "positive" or "neutral" or "negative", "emotion": "<one of the eight emotions>", "confidence": <float 0.0 to 1.0>}'
)


def build_user_msg(title: str, text: str) -> str:
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if text:
        parts.append(f"Review text: {text}")
    body = "\n".join(parts)
    return f"{body}\n\nClassify the sentiment." if body else "(No title or text provided.)\n\nClassify the sentiment."


def classify_one(client: OpenAI, model: str, title: str, text: str,
                 json_mode: bool, max_retries: int):
    """Classify a single review, retrying on transient API failures."""
    user_msg = build_user_msg(title, text)

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            parsed = parse_model_output(raw)
            if parsed["sentiment"]:
                return parsed
            # Model gave something we couldn't parse; retry a few times.
            last_raw = raw.strip()
            empty = not last_raw  # flaky proxies return 200 + empty body under load
        except Exception as e:  # network, auth, rate limit, etc.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429 or (status is None):  # rate-limit or unknown (retry)
                if attempt >= max_retries:
                    return {"sentiment": None, "confidence": None,
                            "error": str(e)[:200]}
                attempt += 1
                time.sleep(min(2 ** attempt, 16))
                continue
            # 400 (bad request / context too long), 401 auth, 404 model...
            return {"sentiment": None, "confidence": None, "error": str(e)[:200]}

        # Empty-body drops are the flakiest case on some OpenAI-compatible proxies:
        # retry more patiently than a plain unparseable reply.
        cap = max_retries + 6 if empty else max_retries
        if attempt >= cap:
            preview = repr(last_raw[:150]) if last_raw else "(empty)"
            return {"sentiment": None, "confidence": None,
                    "error": f"unparseable-model-output: {preview}"}
        attempt += 1
        time.sleep(min(2 ** attempt, 30))


# --------------------------------------------------------------------------- #
#  I/O & orchestration
# --------------------------------------------------------------------------- #
def iter_reviews(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rating_heuristic(rating: float) -> str | None:
    """Proxy ground-truth used only for eval (three classes)."""
    if rating is None:
        return None
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def review_id(r: dict) -> str:
    """Unique review key. asin is the *product* and (user_id, timestamp) alone is
    NOT unique (a user can review two products in the same millisecond), so the
    key includes all three: reviewer + timestamp + product."""
    uid = r.get("user_id"); ts = r.get("timestamp"); a = r.get("asin")
    if uid is not None and ts is not None and a is not None:
        return f"{uid}|{ts}|{a}"
    return str(r.get("asin", "")) or json.dumps(r, sort_keys=True)


def load_processed(output_path: str) -> set:
    """Return review_ids of records that were successfully labeled (label present).
    Failed/null-label rows are deliberately excluded so a resume re-tries them."""
    seen = set()
    if output_path and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("label") and rec.get("review_id"):
                            seen.add(rec["review_id"])
                    except Exception:
                        pass
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Path to .jsonl or .jsonl.gz file")
    ap.add_argument("--out", default="reviews_classified.jsonl",
                    help="Output path [default: reviews_classified.jsonl]")
    ap.add_argument("--base-url", help="OpenAI-compatible base URL (env OPENAI_BASE_URL)")
    ap.add_argument("--api-key", help="API key (env OPENAI_API_KEY; required if not set)")
    ap.add_argument("--model", help="Model name (env OPENAI_MODEL; default gpt-4o-mini)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Classify at most this many reviews")
    ap.add_argument("--sample", type=int, default=None,
                    help="Randomly sample this many reviews (seeded)")
    ap.add_argument("--balanced", type=int, default=None,
                    help="Seeded, class-balanced sample: this many per rating class "
                         "(positive 4-5 / neutral 3 / negative 1-2). Overrides --limit/--sample.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for --sample")
    ap.add_argument("--workers", type=int, default=5,
                    help="Concurrent API requests [default: 5]")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="Retries per review on rate-limit/transient errors [default: 5]")
    ap.add_argument("--json-mode", action="store_true", default=True,
                    help="Request structured JSON output (default on)")
    ap.add_argument("--no-json-mode", action="store_false", dest="json_mode")
    ap.add_argument("--eval", action="store_true",
                    help="Cross-check labels vs star-rating heuristic and print a confusion matrix")
    ap.add_argument("--resume", action="store_true",
                    help="Skip ASINs already written to --out")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        ap.error("No API key. Pass --api-key or set OPENAI_API_KEY.")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url or None)

    # Build the work list.
    reviews = []
    for r in iter_reviews(args.input):
        reviews.append(r)
        if args.limit is not None and args.balanced is None and len(reviews) >= args.limit:
            break

    rng = random.Random(args.seed)
    if args.balanced:
        buckets = {"positive": [], "neutral": [], "negative": []}
        for r in reviews:
            rt = r.get("rating")
            if rt is None:
                continue
            if rt >= 4:
                buckets["positive"].append(r)
            elif rt == 3:
                buckets["neutral"].append(r)
            else:  # <= 2
                buckets["negative"].append(r)
        counts = {g: min(args.balanced, len(buckets[g])) for g in buckets}
        picked = []
        for g in ("positive", "neutral", "negative"):
            picked.extend(rng.sample(buckets[g], counts[g]))
        print(f"[balanced] {counts['positive']} positive / {counts['neutral']} neutral "
              f"/ {counts['negative']} negative  (seed={args.seed})")
        reviews = picked
    elif args.sample is not None and args.sample < len(reviews):
        reviews = rng.sample(reviews, args.sample)

    if args.resume and os.path.exists(args.out):
        seen = load_processed(args.out)
        kept = [r for r in reviews if review_id(r) not in seen]
        print(f"[resume] {len(seen)} already processed, {len(kept)} remaining")
        reviews = kept

    current_ids = {review_id(r) for r in reviews}

    print(f"Target model: {model}")
    print(f"Endpoint base URL: {base_url or 'https://api.openai.com/v1'}")
    print(f"Classifying {len(reviews)} review(s) with {args.workers} worker(s)...")

    counters = {"ok": 0, "failed": 0}
    out_fh = open(args.out, "a", encoding="utf-8") if not args.resume else open(args.out, "a", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(classify_one, client, model,
                      (r.get("title") or "").strip(),
                      (r.get("text") or "").strip(),
                      args.json_mode, args.max_retries): r
            for r in reviews
        }
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            res = fut.result()
            record = {
                "review_id": review_id(r),
                "asin": r.get("asin"),
                "title": (r.get("title") or "").strip(),
                "text": (r.get("text") or "").strip(),
                "rating": r.get("rating"),
                "verified_purchase": r.get("verified_purchase"),
                "label": res["sentiment"],
                "emotion": res.get("emotion"),
                "confidence": res["confidence"],
            }
            if res.get("error"):
                record["error"] = res["error"]
            if record["label"] is None:
                counters["failed"] += 1
            else:
                counters["ok"] += 1
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()
            if i % 25 == 0 or i == len(reviews):
                print(f"  {i}/{len(reviews)}  ok={counters['ok']} failed={counters['failed']}", flush=True)
    out_fh.close()

    # Summary
    summary_lines = [
        f"model={model}",
        f"total={len(reviews)}",
        f"ok={counters['ok']}",
        f"failed={counters['failed']}",
    ]
    if counters["ok"]:
        labels = []
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("review_id") in current_ids and rec.get("label"):
                    labels.append(rec["label"])
        from collections import Counter
        lc = Counter(labels)
        summary_lines.append(f"predicted_positive={lc.get('positive', 0)} "
                             f"predicted_negative={lc.get('negative', 0)}")

    # Optional eval vs rating heuristic (three classes).
    if args.eval:
        from collections import Counter
        cm = Counter()
        n_truth = 0
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("review_id") not in current_ids:
                    continue
                gold = rating_heuristic(rec.get("rating"))
                if gold is None or not rec.get("label"):
                    continue
                n_truth += 1
                cm[(gold, rec["label"])] += 1
        if n_truth:
            classes = ("positive", "neutral", "negative")
            acc = sum(cm[(g, g)] for g in classes) / n_truth
            summary_lines += [f"eval_n={n_truth}", f"eval_accuracy={acc:.4f}"]
            for g in classes:
                tot = sum(cm[(g, p)] for p in classes)
                ok = cm[(g, g)]
                summary_lines.append(f"eval_class_{g}={ok}/{tot}")
            summary_lines.append("confusion(rows=gold, cols=pred):")
            summary_lines.append("      " + "".join(f"{p:>8}" for p in classes))
            for g in classes:
                summary_lines.append(f"{g:>7}" + "".join(f"{cm[(g, p)]:>8}" for p in classes))

    summary_path = args.out.rsplit(".", 1)[0] + ".summary"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    print(f"\nDone. Results -> {args.out}  Summary -> {summary_path}")


if __name__ == "__main__":
    main()
