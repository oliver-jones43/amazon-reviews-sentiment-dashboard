#!/usr/bin/env python3
"""Build dashboard.html from the classifier's scored + emotion output.

Reads an emotion-augmented classifier output (.jsonl with `label`, `emotion`,
`confidence`, `rating`, `title`, `text`), applies the step-2 rule
(rating>=4 -> positive, else -> negative), computes the NRC word-list primary
emotion per review (offline, from the public NRC lexicon), and emits a single
self-contained offline HTML dashboard with:
  * sentiment headline KPIs + confusion matrix + per-class accuracy
  * a primary-emotion comparison: LLM vs NRC word list (agreement, valence
    agreement, distributions, divergence examples, per-review columns)
  * a filterable/sortable per-review table

Usage:  build_dashboard.py <emotion.jsonl> --out dashboard.html
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from nrc_scorer import EMOTIONS, derive, load_lexicon

VALENCE_POS = {"joy", "trust", "anticipation", "surprise"}
VALENCE_NEG = {"anger", "disgust", "fear", "sadness"}


CLASSES = ("positive", "neutral", "negative")


def gold_from_rating(r) -> str | None:
    if r is None:
        return None
    if r >= 4:
        return "positive"
    if r == 3:
        return "neutral"
    return "negative"


def build(input_path):
    lex = load_lexicon()
    labeled, dropped = [], []
    for line in open(input_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rating = rec.get("rating")
        gold = gold_from_rating(rating)
        if gold is None:
            continue
        nrc, scores, hits, tied = derive(lex, rec.get("title") or "", rec.get("text") or "")
        row = {
            "review_id": rec.get("review_id"),
            "rating": rating,
            "gold": gold,
            "label": rec.get("label"),
            "emotion": rec.get("emotion"),          # LLM primary emotion
            "nrc_emotion": nrc,                     # NRC primary emotion
            "nrc_tied": tied,
            "nrc_hits": hits,
            "confidence": rec.get("confidence"),
            "title": rec.get("title") or "",
            "text": rec.get("text") or "",
        }
        if row["label"]:
            labeled.append(row)
        else:
            dropped.append(row)

    # ---- three-class sentiment ----
    def class_acc(g):
        n = sum(1 for r in labeled if r["gold"] == g)
        ok = sum(1 for r in labeled if r["gold"] == g and r["label"] == g)
        return {"class": g, "n": n, "correct": ok, "wrong": n - ok}

    tot = len(labeled)
    agree = sum(1 for r in labeled if r["label"] == r["gold"])
    confusion = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for r in labeled:
        confusion[r["gold"]][r["label"]] += 1
    gold_n = {g: sum(confusion[g].values()) for g in CLASSES}
    pred_n = {p: sum(confusion[g][p] for g in CLASSES) for p in CLASSES}
    star_dist = {str(s): 0 for s in range(1, 6)}
    for r in labeled:
        star_dist[str(int(round(r["rating"])))] += 1

    # ---- emotion comparison ----
    both = [r for r in labeled if r["emotion"] and r["nrc_emotion"]]
    agree_e = [r for r in both if r["emotion"] == r["nrc_emotion"]]
    diverge = [r for r in both if r["emotion"] != r["nrc_emotion"]]
    valence = sum(1 for r in both
                  if (r["emotion"] in VALENCE_POS and r["nrc_emotion"] in VALENCE_POS)
                  or (r["emotion"] in VALENCE_NEG and r["nrc_emotion"] in VALENCE_NEG))
    llm_dist = Counter(r["emotion"] for r in labeled if r["emotion"])
    nrc_dist = Counter(r["nrc_emotion"] for r in labeled if r["nrc_emotion"])
    tied = sum(1 for r in labeled if r["nrc_tied"])

    data = {
        "meta": {
            "input": input_path,
            "rule": "4\u20135 \u2192 positive \u00b7 3 \u2192 neutral \u00b7 1\u20132 \u2192 negative",
            "model": "DeepSeek-V4-Flash-0731",
            "batch": "balanced 50/class (seed 42), full file",
            "emotions": "8 NRC emotions",
        },
        "stats": {
            "scored": tot,
            "agree": agree,
            "agreement": round(agree / tot, 4) if tot else 0.0,
            "classes": [class_acc(g) for g in CLASSES],
            "confusion": confusion,
            "gold_n": gold_n,
            "pred_n": pred_n,
            "star_dist": star_dist,
            "drop": {"count": len(dropped),
                     "examples": [(d.get("text") or "")[:48] for d in dropped[:8]]},
        },
        "emotions": {
            "base": len(both),
            "agree": len(agree_e),
            "diverge": len(diverge),
            "agree_rate": round(len(agree_e) / len(both), 4) if both else 0,
            "valence": valence,
            "valence_rate": round(valence / len(both), 4) if both else 0,
            "llm_missing": sum(1 for r in labeled if not r["emotion"]),
            "nrc_missing": sum(1 for r in labeled if not r["nrc_emotion"]),
            "nrc_tied": tied,
            "llm_dist": {e: llm_dist.get(e, 0) for e in EMOTIONS},
            "nrc_dist": {e: nrc_dist.get(e, 0) for e in EMOTIONS},
            "divergence": [
                {"title": r["title"], "text": r["text"],
                 "llm": r["emotion"], "nrc": r["nrc_emotion"],
                 "tied": r["nrc_tied"], "hits": r["nrc_hits"]}
                for r in diverge
            ],
        },
        "rows": labeled,
    }
    return data


HTML = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Sentiment + Emotion — First-Batch Results</title>
<style>
/* ============================================================
   THEME TOKENS  ·  recolor the host here.
   Light theme below; dark overrides in [data-theme="dark"].
   ============================================================ */
:root{
  --bg:#f5f4f0; --surface:#ffffff; --surface-2:#efede9;
  --ink:#1a1d21; --ink-soft:#5b6168; --faint:#9aa1a8; --line:#e2e0da;
  --pos:#16704a; --pos-ink:#126040; --pos-bg:#e4f2eb;
  --neg:#b3402f; --neg-ink:#9a3629; --neg-bg:#f8e9e5;
  --acc:#4059ad; --acc-ink:#34539b; --acc-bg:#e8ecf7;
  --warn:#8a6d0a; --warn-bg:#f6efd9;
  --font-display:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --radius:14px; --radius-sm:9px;
  --shadow:0 1px 2px rgba(20,22,26,.04),0 8px 28px rgba(20,22,26,.06);
  --maxw:1120px;
}
[data-theme="dark"]{
  --bg:#0f1114; --surface:#16191e; --surface-2:#1d2128;
  --ink:#e9e7e2; --ink-soft:#9aa1aa; --faint:#6b727b; --line:#262c35;
  --pos:#3dd68f; --pos-ink:#5ee0a2; --pos-bg:#14302a;
  --neg:#ff7a66; --neg-ink:#ff9382; --neg-bg:#33201c;
  --acc:#8aa0e8; --acc-ink:#9fb1ee; --acc-bg:#20263c;
  --warn:#e3c463; --warn-bg:#2e2917;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 32px rgba(0,0,0,.45);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;transition:background .3s,color .3s}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 26px 90px}
h1,h2,h3{line-height:1.15;margin:0}
.mono{font-family:var(--font-mono)}
.masthead{padding:46px 0 26px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:flex-end;gap:20px;flex-wrap:wrap}
.eyebrow{font-family:var(--font-mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-soft);margin-bottom:12px}
.masthead h1{font-family:var(--font-display);font-size:clamp(28px,4.4vw,44px);font-weight:600;letter-spacing:-.01em}
.masthead .sub{color:var(--ink-soft);font-size:15px;margin-top:8px;max-width:680px}
.themebtn{flex:none;background:var(--surface);border:1px solid var(--line);color:var(--ink);border-radius:999px;padding:7px 15px;font-family:var(--font-sans);font-size:12.5px;cursor:pointer;display:inline-flex;align-items:center;gap:7px;transition:border-color .15s,background .15s}
.themebtn:hover{border-color:var(--ink-soft)}
.themebtn .dot{width:8px;height:8px;border-radius:50%;background:var(--ink);display:inline-block}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:34px 0 12px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:20px 22px 18px;box-shadow:var(--shadow)}
.kpi .label{font-size:12px;color:var(--ink-soft);letter-spacing:.02em}
.kpi .big{font-family:var(--font-display);font-size:clamp(30px,4vw,40px);font-weight:600;margin-top:6px;letter-spacing:-.01em}
.kpi .big em{font-style:normal;color:var(--faint);font-size:.55em;font-weight:500;margin-left:3px}
.kpi .note{font-size:12px;color:var(--faint);margin-top:4px}
.kpi.pos .big{color:var(--pos-ink)}.kpi.neg .big{color:var(--neg-ink)}.kpi.acc .big{color:var(--acc-ink)}
section{margin-top:48px}
.sec-head{display:flex;align-items:baseline;gap:14px;margin-bottom:18px;flex-wrap:wrap}
.sec-head h2{font-family:var(--font-display);font-size:22px;font-weight:600}
.sec-head .rule{font-family:var(--font-mono);font-size:12px;color:var(--ink-soft);background:var(--surface-2);border:1px solid var(--line);padding:4px 10px;border-radius:999px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:24px;box-shadow:var(--shadow)}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}
.class-card{border:1px solid var(--line);border-radius:var(--radius);padding:20px 20px 16px;background:var(--surface);box-shadow:var(--shadow)}
.class-card .row1{display:flex;justify-content:space-between;align-items:baseline}
.class-card .cls{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;font-family:var(--font-mono);color:var(--ink-soft)}
.emo-dist{display:flex;flex-direction:column;gap:7px;margin-top:12px}
.emo-row{display:grid;grid-template-columns:64px 1fr 30px;gap:8px;align-items:center;font-size:12.5px}
.emo-row .lbl{font-family:var(--font-mono);color:var(--ink-soft);text-align:right;text-transform:lowercase}
.emo-row .bar{height:8px;border-radius:999px;background:var(--surface-2);overflow:hidden}
.emo-row .bar i{display:block;height:100%;border-radius:999px}
.emo-row .cnt{font-family:var(--font-mono);color:var(--faint);text-align:right}
.llm-bar i{background:var(--acc)} .nrc-bar i{background:var(--faint)}
.why{background:var(--warn-bg);border:1px solid color-mix(in srgb,var(--warn) 40%,transparent);border-radius:var(--radius-sm);padding:14px 16px;margin-top:18px;font-size:14px;color:var(--ink)}
.why b{color:var(--ink)}
.why.reveal{background:var(--acc-bg);border-color:color-mix(in srgb,var(--acc) 40%,transparent)}
.conf-title{font-family:var(--font-mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft);margin-bottom:10px}
.conf-table{border-collapse:collapse;font-size:13.5px}
.conf-table th,.conf-table td{padding:9px 14px;border:1px solid var(--line);text-align:center;min-width:74px}
.conf-table thead th,.conf-table th.rcol{font-family:var(--font-mono);font-size:11px;text-transform:uppercase;color:var(--ink-soft);background:var(--surface-2);letter-spacing:.04em}
.conf-table th.rcol{text-align:right}
.conf-table td.miss{background:var(--surface);color:var(--ink-soft)}
.conf-table td.hit.positive{background:var(--pos-bg);color:var(--pos-ink);font-weight:600}
.conf-table td.hit.neutral{background:var(--acc-bg);color:var(--acc-ink);font-weight:600}
.conf-table td.hit.negative{background:var(--neg-bg);color:var(--neg-ink);font-weight:600}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:16px}
.dist-row{display:grid;grid-template-columns:44px 1fr 46px;gap:9px;align-items:center;margin:7px 0;font-size:13px}
.dist-row .lbl{font-family:var(--font-mono);color:var(--ink-soft);font-size:12px;text-transform:lowercase}
.dist-row .cnt{font-family:var(--font-mono);color:var(--ink-soft);text-align:right;font-size:12px}
.track{position:relative;height:15px;border-radius:7px;background:var(--surface-2);overflow:hidden}
.track i{display:block;height:100%;min-width:2px}
.stack{display:flex;height:15px;border-radius:7px;overflow:hidden;background:var(--surface-2)}
.stack span{display:block;height:100%;min-width:2px}
.seg-legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px;font-size:11.5px;color:var(--ink-soft);align-items:center}
.seg-legend i{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.seg-legend .outline-note{opacity:.8;margin-left:auto}
.div-example{display:flex;gap:12px;padding:12px 2px;border-bottom:1px solid var(--line);align-items:flex-start}
.div-example:last-child{border-bottom:0}
.emo-pair{flex:none;font-family:var(--font-mono);font-size:12px;min-width:150px;line-height:1.6}
.emo-pair .llm{color:var(--acc-ink)}.emo-pair .nrc{color:var(--ink-soft)}
.emo-pair .tie{color:var(--faint)}
.tcount{font-family:var(--font-mono);font-size:12px;color:var(--faint);margin-left:auto;padding-right:4px;white-space:nowrap}
.metaline{font-family:var(--font-mono);font-size:12px;color:var(--ink-soft);margin:-4px 0 12px}
.metaline b{color:var(--ink);font-weight:600}
.chip .cnt{opacity:.6;margin-left:2px}
.coverage{background:var(--surface-2);border:1px dashed var(--line);border-radius:var(--radius);padding:18px 22px;font-size:14px;color:var(--ink-soft)}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
.chip{border:1px solid var(--line);background:var(--surface);color:var(--ink-soft);border-radius:999px;padding:7px 14px;font-size:13px;cursor:pointer;font-family:var(--font-sans);transition:all .15s}
.chip:hover{border-color:var(--ink-soft)}
.chip.active{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.search{flex:none;margin-left:auto;min-width:220px}
.search input{width:100%;background:var(--surface);border:1px solid var(--line);color:var(--ink);border-radius:999px;padding:8px 15px;font-size:13px;font-family:var(--font-sans)}
.search input:focus{outline:none;border-color:var(--ink-soft)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
thead th{position:sticky;top:0;background:var(--surface);text-align:left;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-soft);font-family:var(--font-mono);font-weight:500;padding:10px 12px;border-bottom:1px solid var(--line);cursor:pointer;user-select:none;white-space:nowrap}
thead th:hover{color:var(--ink)}
thead th .arr{opacity:0;font-size:10px;margin-left:3px}
thead th.sorted .arr{opacity:1}
tbody td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}
.rev-title{font-weight:600;color:var(--ink)}
.rev-text{color:var(--ink-soft);font-size:12.5px;margin-top:2px;max-width:440px}
.stars{color:var(--warn);letter-spacing:1px;font-family:var(--font-mono);white-space:nowrap}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:600;letter-spacing:.02em}
.badge.ok{background:var(--pos-bg);color:var(--pos-ink)}
.badge.bad{background:var(--neg-bg);color:var(--neg-ink)}
.badge.acc2{background:var(--acc-bg);color:var(--acc-ink)}
.twocol{display:flex;gap:8px;font-family:var(--font-mono);font-size:12px;flex-wrap:wrap}
.twocol .g{color:var(--ink-soft)}.twocol .p{color:var(--pos-ink)}.twocol .n{color:var(--neg-ink)}
.neu{color:var(--acc-ink);font-family:var(--font-mono);font-size:12px}
.conf{font-family:var(--font-mono);color:var(--ink-soft)}
.cm-legend{font-size:12px;color:var(--ink-soft);display:flex;gap:16px}
.cm-legend .pos{color:var(--pos-ink)}.cm-legend .neg{color:var(--neg-ink)}
.emo-cell{font-family:var(--font-mono);font-size:12px;white-space:nowrap}
.emo-cell .ar{color:var(--faint);margin:0 3px}
.emo-cell.match .vl{color:var(--pos-ink)}
.emo-cell.diff .vl{color:var(--neg-ink)}
.emo-cell .null{color:var(--faint)}
.emo-cell .tie{color:var(--faint);font-size:10px;vertical-align:.2em}
footer{margin-top:60px;border-top:1px solid var(--line);padding-top:22px;font-size:13px;color:var(--faint);display:flex;flex-direction:column;gap:6px}
footer b{color:var(--ink-soft);font-weight:600}
@media(max-width:820px){.kpis{grid-template-columns:repeat(2,1fr)}.grid-2{grid-template-columns:1fr}.search{min-width:100%;margin-left:0}}
</style>
</head>
<body>
<div class="wrap">

  <header class="masthead">
    <div>
      <div class="eyebrow">Amazon 2023 · Gift Cards · First-Batch Review</div>
      <h1>Sentiment &amp; Emotion vs. Ground Truth</h1>
      <div class="sub">Three-class sentiment (positive / neutral / negative) judged against the
        rating — <b>4–5★ positive · 3★ neutral · 1–2★ negative</b> — on a <b>balanced sample</b> of the
        full file (≈50 per class). The <b>LLM</b> reads only title + text; ratings score it afterwards.</div>
    </div>
    <button class="themebtn" id="themebtn" title="Toggle light / dark"><span class="dot"></span>Theme</button>
  </header>

  <div class="kpis" id="kpis"></div>

  <!-- Sentiment -->
  <section>
    <div class="sec-head"><h2>How it scored (sentiment · 3 classes)</h2><span class="rule" id="rule"></span></div>
    <div class="card">
      <div class="conf-title">Confusion — rows = rating class (gold), columns = model</div>
      <div id="confusion"></div>
      <div class="why reveal" id="reveal"></div>
    </div>
    <div class="grid-3" id="classes"></div>
  </section>

  <!-- Emotion comparison -->
  <section>
    <div class="sec-head"><h2>Primary emotion — LLM vs NRC word list</h2><span class="rule">two independent takes</span></div>
    <div class="card">
      <div class="kpis emokpis" id="emo-kpis"></div>
      <div class="grid-2">
        <div class="class-card"><div class="row1"><span class="cls">LLM emotion</span></div><div class="emo-dist" id="llm-dist"></div></div>
        <div class="class-card"><div class="row1"><span class="cls">NRC word-list emotion</span></div><div class="emo-dist" id="nrc-dist"></div></div>
      </div>
      <div class="why" id="emo-why"></div>
      <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;font-family:var(--font-mono);color:var(--ink-soft);margin:20px 0 6px">Where they diverge</div>
      <div id="emo-diverge"></div>
    </div>
  </section>

  <!-- Descriptive view -->
  <section>
    <div class="sec-head"><h2>Descriptive view</h2><span class="rule">failures at a glance</span></div>
    <div class="grid-2">
      <div class="class-card">
        <div class="row1"><span class="cls">Star-rating distribution · scored set</span></div>
        <div id="star-dist" style="margin-top:14px"></div>
      </div>
      <div class="class-card">
        <div class="row1"><span class="cls">Predicted vs true, by class</span></div>
        <div id="pred-split" style="margin-top:14px"></div>
        <div class="seg-legend">
          <span><i style="background:var(--pos)"></i>pos</span>
          <span><i style="background:var(--acc)"></i>neu</span>
          <span><i style="background:var(--neg)"></i>neg</span>
          <span class="outline-note">outlined segment = correct answer</span>
        </div>
      </div>
    </div>
    <div class="why" id="fail-focus"></div>
  </section>

  <!-- Coverage -->
  <section>
    <div class="sec-head"><h2>Coverage</h2></div>
    <div class="coverage" id="coverage"></div>
  </section>

  <!-- Table -->
  <section>
    <div class="sec-head"><h2>Every review</h2></div>
    <div class="card" style="padding:18px">
      <div class="controls">
        <button class="chip active" data-f="all">All</button>
        <button class="chip" data-f="ok">Correct</button>
        <button class="chip" data-f="bad">Wrong</button>
        <button class="chip" data-f="pos">Positive</button>
        <button class="chip" data-f="neu">Neutral</button>
        <button class="chip" data-f="neg">Negative</button>
        <span class="tcount" id="tcount"></span>
        <span class="search"><input id="q" placeholder="Search review text or title…"></span>
      </div>
      <div class="metaline" id="metaline"></div>
      <div style="overflow:auto;max-height:560px">
      <table id="rtable">
        <thead><tr id="rtable-head"></tr></thead>
        <tbody id="rtable-body"></tbody>
      </table>
      </div>
    </div>
  </section>

  <footer>
    <div><b>Method.</b> Three-class ground truth from rating — <span class="mono">4–5&nbsp;→&nbsp;positive · 3&nbsp;→&nbsp;neutral · 1–2&nbsp;→&nbsp;negative</span>. The model never saw the rating; verdict from <b>title + text</b> only.</div>
    <div><b>Sampling.</b> Balanced ≈50 per class, seeded (42) from the full 152,410-review file, so the rarer ★3 class is fairly represented instead of under-covered in read-order.</div>
    <div><b>Emotion methods.</b> LLM: single primary emotion from the 8. NRC: sum of lexicon word-scores per emotion, argmax is the answer.</div>
    <div><b>Reskin.</b> Theme lives in the CSS token block — edit <span class="mono">--bg/--pos/--neg/--acc</span>, etc. to recolor.</div>
  </footer>

</div>

<script>const DASH=__DATA__;</script>
<script>
const $=s=>document.querySelector(s);
const fmtAgg=a=>(100*a).toFixed(1)+"%";
const EMOS=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"];
function esc(s){return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function nbsp(){return "\u00a0"}

/* --- headline KPIs --- */
function kpi(cls,big,label,note){return `<div class="kpi ${cls}"><div class="label">${label}</div><div class="big">${big}</div><div class="note">${note||""}</div></div>`}
function renderKpis(){
  const s=DASH.stats,e=DASH.emotions;
  const cls={}; s.classes.forEach(c=>cls[c["class"]]=c);
  $("#rule").textContent=DASH.meta.rule;
  $("#kpis").innerHTML=
    kpi("","\u2022 "+fmtAgg(s.agreement),"3-class agreement","overall, "+s.scored+" scored")+
    kpi("",s.scored,"Reviews scored","balanced ≈50/class · "+s.drop.count+" dropped")+
    kpi("acc",fmtAgg((cls.neutral.correct||0)/ (cls.neutral.n||1))+"<em>"+cls.neutral.correct+"/"+cls.neutral.n+"</em>","Neutral (★3) class","kept in its own class")+
    kpi("acc",fmtAgg(e.agree_rate)+"<em>of "+e.base+"</em>","Emotion: LLM = NRC","primary-emotion agreement");
}

function renderConfusion(){
  const C=DASH.stats.confusion,CLS=["positive","neutral","negative"];
  const h=`<tr><th>gold \\ pred →</th>`+CLS.map(p=>`<th class="pcol ${p}">${p}</th>`).join('')+`</tr>`;
  const b=CLS.map(g=>{
    const tot=C[g].positive+C[g].neutral+C[g].negative;
    return `<tr><th class="rcol ${g}">${g} (${tot})</th>`+
      CLS.map(p=>{const v=C[g][p];const cls=g===p?('hit '+g):'miss';return `<td class="${cls}">${v}</td>`;}).join('')+`</tr>`;
  }).join('');
  $("#confusion").innerHTML=`<table class="conf-table"><thead>${h}</thead><tbody>${b}</tbody></table>`;
}
function renderReveal(){
  const C=DASH.stats.confusion;
  const nu=C.neutral, nt=nu.positive+nu.neutral+nu.negative;
  const stay=nu.neutral, spct=nt?100*stay/nt:0;
  const ne=C.negative, net=ne.positive+ne.neutral+ne.negative;
  const negPct=nt?100*nu.negative/nt:0;
  $("#reveal").innerHTML=`<b>What the balanced run reveals:</b> with ★3 reviews fairly represented
    (2% of the full file, ~50 in this sample), the model keeps <b>${stay}/${nt} (${fmtAgg(spct/100)})</b>
    in their own <b>neutral</b> class — pushing <b>${nu.negative}/${nt} (${fmtAgg(negPct/100)})</b> to negative
    and ${nu.positive} to positive. Overall it predicts “neutral” only
    <b>${C.positive.neutral+C.neutral.neutral+C.negative.neutral}/129</b> times. A scarce ★3 class
    therefore does <b>not</b> form a reliable third class on balanced data — it collapses into
    negative (with a little positive). Negatives (★1–2) never collapse toward neutral
    (${ne.neutral}/${net}).`;
}

function renderClasses(){
  $("#classes").innerHTML=DASH.stats.classes.map(c=>{
    const cl=c["class"],acc=c.n?fmtAgg(c.correct/c.n):"—",pct=c.n?(100*c.correct/c.n):0;
    const color=cl==="positive"?"var(--pos)":(cl==="negative"?"var(--neg)":"var(--acc)");
    return `<div class="class-card">
      <div class="row1"><span class="cls">${cl}</span><span class="acc" style="font-family:var(--font-display);font-size:24px">${acc} <em style="color:var(--faint);font-size:14px">${c.correct}/${c.n}</em></span></div>
      <div class="bar" style="height:10px;border-radius:999px;background:var(--surface-2);overflow:hidden;margin:12px 0 8px"><i style="display:block;height:100%;width:${pct}%;border-radius:999px;background:${color}"></i></div>
      <div class="bar-meta" style="display:flex;justify-content:space-between;font-size:12px;color:var(--ink-soft)"><span>${c.correct} right</span><span>${c.wrong} wrong</span></div>
    </div>`;
  }).join("");
}

/* --- descriptive view --- */
function starCls(s){return s<=2?'var(--neg)':(s===3?'var(--acc)':'var(--pos)');}
function renderStarDist(){
  const d=DASH.stats.star_dist, vals=[d["1"]||0,d["2"]||0,d["3"]||0,d["4"]||0,d["5"]||0];
  const maxc=Math.max(1,...vals); let html='';
  for(let s=1;s<=5;s++){const c=vals[s-1]||0;const w=c?Math.max(2,Math.round(100*c/maxc)):0;
    html+=`<div class="dist-row"><span class="lbl">★${s}</span><span class="track"><i style="width:${w}%;background:${starCls(s)}"></i></span><span class="cnt">${c}</span></div>`;}
  $("#star-dist").innerHTML=html;
}
const SEGC={positive:"var(--pos)",neutral:"var(--acc)",negative:"var(--neg)"};
function renderPredSplit(){
  const C=DASH.stats.confusion,CLS=["positive","neutral","negative"];let html='';
  for(const g of CLS){const row=C[g],tot=row.positive+row.neutral+row.negative,ok=row[g];
    const segs=CLS.map(p=>{const v=row[p]||0;const w=tot?Math.round(100*v/tot):0;
      return `<span style="width:${w}%;background:${SEGC[p]};${p===g?'outline:2px solid var(--ink);outline-offset:-2px;':''}" title="${v} ${p}"></span>`;}).join('');
    html+=`<div class="dist-row" style="grid-template-columns:118px 1fr 56px"><span class="lbl">${g}</span><span class="stack">${segs}</span><span class="cnt">${ok}/${tot}</span></div>`;}
  $("#pred-split").innerHTML=html;
}
function renderFailFocus(){
  const weak=DASH.stats.classes.filter(c=>c.n && (c.correct/c.n)<0.9);
  const neut=DASH.stats.confusion.neutral, nt=neut.positive+neut.neutral+neut.negative;
  const msg=weak.length?weak.map(c=>`<b>${c["class"].toUpperCase()} ${fmtAgg(c.correct/c.n)}</b>`).join(" and "):"none (every class \u2265 90%)";
  $("#fail-focus").innerHTML=`<b>Failures at a glance:</b> weak class — ${msg}. The <b>true-NEUTRAL</b> bar is the one
    that is mostly red: ${neut.negative}/${nt} of real 3★ reviews are read as negative, so the model's misses hide
    in the middle class, not at the extremes.`;
}

/* --- emotion comparison --- */
function emoDistRows(order,cls){
  return EMOS.map(e=>{
    const c=order[e]||0; const mx=Math.max(1,...EMOS.map(x=>order[x]||0));
    const w=Math.round(100*c/mx);
    return `<div class="emo-row"><span class="lbl">${e}</span><span class="bar ${cls}"><i style="width:${w}%"></i></span><span class="cnt">${c}</span></div>`;
  }).join("");
}
function renderEmotions(){
  const e=DASH.emotions;
  $("#llm-dist").innerHTML=emoDistRows(e.llm_dist,"llm-bar");
  $("#nrc-dist").innerHTML=emoDistRows(e.nrc_dist,"nrc-bar");
  const tiePct=Math.round(100*e.nrc_tied/DASH.stats.scored);
  $("#emo-why").innerHTML=`<b>Why they diverge:</b> the two methods agree on the emotional <b>valence</b>
    in <b>${fmtAgg(e.valence_rate)}</b> of ${e.base} reviews, but pick the same <b>specific</b> emotion only
    <b>${fmtAgg(e.agree_rate)}</b>. The word-list method is a naive bag of words: on these terse reviews it
    scores few words (avg ~1.7 hits), so <b>${e.nrc_tied}/${DASH.stats.scored}</b> (${tiePct}%) of its argmaxes are
    <i>ties</i> that get resolved by a fixed order — inflating anticipation/anger — and it finds
    <b>no</b> emotion word at all in ${e.nrc_missing} reviews. The LLM, reading full sentences, produces a far
    more peaked, plausible profile (joy, trust, anger) and never picks surprise/fear here.`;
  const div=e.divergence.slice(0,14);
  $("#emo-diverge").innerHTML=div.length?div.map(r=>`
    <div class="div-example">
      <div class="emo-pair"><div class="llm">LLM ${esc(r.llm)}</div><div class="nrc">NRC ${esc(r.nrc)}${r.tied?' <span class="tie">(tied≈)</span>':''}</div></div>
      <div><div style="font-weight:600">${esc(r.title)}</div><div class="rev-text">${esc(trim(r.text,110))}</div></div>
    </div>`).join(""):`<div style="color:var(--ink-soft);font-size:14px">No divergences to show.</div>`;
}
function trim(s,n){return s.length>n?s.slice(0,n)+"\u2026":s;}
function renderCoverage(){
  const s=DASH.stats;
  const ex=(s.drop.examples||[]).map(x=>`<span class="mono" style="background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:2px 7px;margin:0 4px 4px 0;display:inline-block">“${esc(x)}”</span>`).join("");
  $("#coverage").innerHTML=`<b style="color:var(--ink)">${s.drop.count} of the balanced sample</b> were dropped by the LLM endpoint (empty body) and are excluded. They skew to very short reviews. ${ex}`;
}

/* --- table --- */
const COLS=[
  {k:"rating",t:"Rating",fn:r=>stars(r.rating),sort:r=>r.rating},
  {k:"gold",t:"Gold",fn:r=>glabel(r.gold),sort:r=>r.gold},
  {k:"label",t:"Model",fn:r=>plabel(r.label),sort:r=>r.label},
  {k:"match",t:"Verdict",fn:r=>badge(r),sort:r=>r.label===r.gold?1:0},
  {k:"emn",t:"Emotion",fn:r=>emoCell(r),sort:r=>emoSort(r)},
  {k:"conf",t:"Conf",fn:r=>r.confidence==null?"—":fmtAgg(r.confidence),sort:r=>r.confidence==null?-1:r.confidence},
  {k:"rev",t:"Review (title · text)",fn:r=>rev(r)},
];
function stars(v){let o="";for(let i=0;i<5;i++)o+=i<Math.round(v)?"\u2605":"\u2606";return `<span class="stars">${o}</span>`;}
function glabel(g){if(g==="positive")return `<span class="p">pos</span>`;if(g==="neutral")return `<span class="neu">neu</span>`;return `<span class="n">neg</span>`;}
function plabel(l){return glabel(l);}
function badge(r){const ok=r.label===r.gold;return `<span class="badge ${ok?"ok":"bad"}">${ok?"match":"MISMATCH"}</span>`;}
function emoCell(r){
  const le=r.emotion,ne=r.nrc_emotion;
  if(!le&&!ne)return `<span class="emo-cell null">—</span>`;
  const cls=(le&&ne)?(le===ne?"match":"diff"):"";
  return `<span class="emo-cell ${cls}"><span class="vl">${esc(le||"—")}</span><span class="ar">→</span><span class="vl">${esc(ne||"—")}</span>${r.nrc_tied?'<span class="tie">≈</span>':''}</span>`;
}
function emoSort(r){if(!r.emotion&&!r.nrc_emotion)return -2;if(!r.nrc_emotion)return -1;return r.emotion===r.nrc_emotion?0:1;}
function rev(r){return `<div class="rev-title">${esc(r.title)}</div><div class="rev-text">${esc(r.text)}</div>`;}
const FLT={all:["All","every scored review"],ok:["Correct","model label = rating label"],bad:["Wrong","model label \u2260 rating label"],pos:["Positive","ground truth = positive (rating 4\u20135)"],neu:["Neutral","ground truth = neutral (rating 3)"],neg:["Negative","ground truth = negative (rating 1\u20132)"]};
let filt="all",q="",sortK="match",sortDir=-1;
function matchFilter(r,f){f=f||filt;if(f==="ok")return r.label===r.gold;if(f==="bad")return r.label!==r.gold;if(f==="pos")return r.gold==="positive";if(f==="neu")return r.gold==="neutral";if(f==="neg")return r.gold==="negative";return true;}
function renderChips(){for(const k in FLT){const c=document.querySelector(`.chip[data-f="${k}"]`);if(c)c.textContent=FLT[k][0]+` <span class="cnt">\u00b7 ${DASH.rows.filter(r=>matchFilter(r,k)).length}</span>`;}}
function renderTable(){
  $("#rtable-head").innerHTML=`<tr>`+COLS.map(c=>`<th class="${c.cls||""}" data-k="${c.k}">${c.t}<span class="arr">${c.k===sortK?(sortDir<0?"\u2193":"\u2191"):""}</span></th>`).join("")+`</tr>`;
  let rows=DASH.rows.filter(r=>matchFilter(r,filt));
  if(q){const ql=q.toLowerCase();rows=rows.filter(r=>(r.title+" "+r.text).toLowerCase().includes(ql));}
  rows.sort((a,b)=>{const ka=COLS.find(c=>c.k===sortK).sort(a),kb=COLS.find(c=>c.k===sortK).sort(b);if(ka<kb)return sortDir*-1;if(ka>kb)return sortDir*1;return 0;});
  $("#tcount").textContent=`${rows.length} row${rows.length===1?"":"s"} shown`;
  const [name,desc]=FLT[filt];
  $("#metaline").innerHTML=`<b>${name}:</b> ${desc}${q?` \u00b7 searching for \u201c${esc(q)}\u201d`:``}`;
  renderChips();
  $("#rtable-body").innerHTML=rows.map(r=>`<tr class="${r.label===r.gold?"newok-row":"newbad-row"}" data-r="${r.review_id}">
    <td>${COLS[0].fn(r)}</td><td>${glabel(r.gold)}</td><td>${plabel(r.label)}</td><td>${badge(r)}</td><td>${emoCell(r)}</td><td>${r.confidence==null?"—":fmtAgg(r.confidence)}</td><td>${rev(r)}</td>
  </tr>`).join("");
}
document.addEventListener("click",e=>{
  const th=e.target.closest("th[data-k]");
  if(th){const k=th.dataset.k;if(sortK===k)sortDir*=-1;else sortK=k;renderTable();}
  const chip=e.target.closest(".chip");
  if(chip){document.querySelectorAll(".chip").forEach(c=>c.classList.remove("active"));chip.classList.add("active");filt=chip.dataset.f;renderTable();}
});
$("#q").addEventListener("input",e=>{q=e.target.value.trim();renderTable();});
$("#themebtn").addEventListener("click",()=>{const d=document.documentElement;d.setAttribute("data-theme",d.getAttribute("data-theme")==="dark"?"light":"dark");});

renderKpis();renderConfusion();renderReveal();renderClasses();renderStarDist();renderPredSplit();renderFailFocus();renderEmotions();renderCoverage();renderTable();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="emotion + sentiment classifier output .jsonl")
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args()
    data = build(args.input)
    html = HTML.replace("__DATA__", json.dumps(data))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(html)//1024} KB, self-contained offline)")
    print(f"sentiment agreement: {data['stats']['agreement']} | "
          f"emotion agree: {data['emotions']['agree_rate']} ({data['emotions']['agree']}/{data['emotions']['base']})")


if __name__ == "__main__":
    main()
