# Prompt used for LLM scoring

The LLM scoring script (`classify_reviews.py`) sends each review two messages:
a fixed **system** prompt and a **user** request containing only the review's
title and body text (it never sees the rating).

## System prompt

```
You are a sentiment + emotion annotator for product reviews. Based only on the
title and body text of each review:
(1) classify the overall sentiment into EXACTLY one of THREE classes:
    POSITIVE (praising, satisfied, recommending), NEUTRAL (mixed, balanced,
    informational, or flatly factual — neither clearly positive nor negative), or
    NEGATIVE (critical, disappointed, complaining).
(2) choose the single PRIMARY emotion expressed, from EXACTLY one of these eight:
    anger, anticipation, disgust, fear, joy, sadness, surprise, trust.
Reply with ONLY valid JSON matching exactly:
{"sentiment": "positive" or "neutral" or "negative",
 "emotion": "<one of the eight emotions>",
 "confidence": <float 0.0 to 1.0>}
```

## User request (per review)

```
Title: <review title>
Review text: <review body text>

Classify the sentiment.
```

## Response contract

The model returns a single JSON object:

```json
{
  "sentiment": "positive | neutral | negative",
  "emotion": "anger | anticipation | disgust | fear | joy | sadness | surprise | trust",
  "confidence": 0.0 .. 1.0
}
```

The scorer parses this tolerantly (JSON object first, then a bare token
fallback) and validates `sentiment` and `emotion` against the allowed values.
Any review whose reply can't be parsed is flagged as **failed** and never
silently guessed.
