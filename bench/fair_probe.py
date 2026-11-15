#!/usr/bin/env python3
"""Fair prefill: what a long prompt's read does to everyone else (our own method; the other team's bench_fairness.py
asks the same questions). K decoding streams start first (512-token replies, different prompts); after 2 s a long
prompt (--long tokens, one reply token); after --delay s more a short chat request (--short-tokens). Streamed, the
client's clock.

    fair_probe.py --url http://127.0.0.1:8000 --long 131072 --others 4 --label L [--log F] [--delay 3]

One line:
  fair_probe label=L long_tokens=.. long_ttft_s=.. long_prefill_tps=.. short_first_token_s=.. short_tps=..
             others=K others_tps_inside=.. others_tps_outside=.. others_gap_max_s=.. others_gap_p95_s=..
(others_chunks_inside_per_s: the decoders' streamed chunks (a round's tokens each) a second, all of them together,
over their decoding time inside the window from the long prompt's start to its first token; outside: over their
decoding time before and after it (nan when they ended inside it); gap: the largest wait between two of a decoder's
chunks inside the window.)
Standard library only; requests go to --url alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from longctx import OVERHEAD, PER_LINE, document  # noqa: E402


def stream(url, model, content, tokens, t_ref, out, key, thinking=False):
    body = {"model": model, "max_tokens": tokens, "temperature": 0, "stream": True, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": thinking}, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, times, n = time.perf_counter(), [], 0
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    n = d["usage"].get("completion_tokens") or n
                for ch in d.get("choices") or []:
                    de = ch.get("delta") or {}
                    if de.get("content") or de.get("reasoning_content"):
                        times.append(time.perf_counter() - t_ref)
    except Exception as exc:  # noqa: BLE001
        out[key] = {"error": str(exc)[:200]}
        return
    out[key] = {"start": t0 - t_ref, "times": times, "tokens": n, "end": time.perf_counter() - t_ref}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--long", type=int, default=131072)
    p.add_argument("--others", type=int, default=4)
    p.add_argument("--short-tokens", type=int, default=200)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="fair")
    p.add_argument("--log", default="")
    a = p.parse_args()
    with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    T0 = time.perf_counter()
    res: dict = {}
    olds = [threading.Thread(target=stream, args=(a.url, model, f"Write a long essay about topic number {i}, part one "
                                                  f"(variant {a.seed}).", 2000, T0, res, f"old{i}"))
            for i in range(a.others)]
    for t in olds:
        t.start()
    time.sleep(2.0)
    lines = max(1, round((a.long - OVERHEAD) / PER_LINE))
    long_doc = document(lines, 9000 + a.seed) + "\n\nReply with one word."
    long_res: dict = {}

    def run_long():
        t = time.perf_counter()
        body = {"model": model, "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": long_doc}]}
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
        long_res.update(start=t - T0, end=time.perf_counter() - T0, tokens=d["usage"]["prompt_tokens"],
                        cached=d["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0))

    th = threading.Thread(target=run_long)
    th.start()
    time.sleep(a.delay)
    stream(a.url, model, "Explain in a few sentences what photosynthesis is.", a.short_tokens, T0, res, "new")
    th.join()
    for t in olds:
        t.join()
    lo, hi = long_res["start"], long_res["end"]
    new = res.get("new") or {}
    inside = outside = 0
    in_span = out_span = 0.0
    gaps = []
    for i in range(a.others):
        r = res.get(f"old{i}") or {}
        ts = r.get("times") or []
        if not ts:
            continue
        inside += sum(lo <= t <= hi for t in ts)
        outside += sum(t < lo or t > hi for t in ts)
        gaps += [b - x for x, b in zip(ts, ts[1:]) if lo <= x <= hi]
        first, last = ts[0], ts[-1]                      # this decoder's own decoding time, in and out of the window
        both = max(0.0, min(last, hi) - max(first, lo))
        in_span += both
        out_span += max(0.0, last - first - both)
    gaps.sort()
    line = (f"fair_probe label={a.label} long_tokens={long_res['tokens']} long_cached={long_res['cached']} "
            f"long_ttft_s={hi - lo:.2f} long_prefill_tps={(long_res['tokens'] - long_res['cached']) / (hi - lo):.0f} "
            f"short_first_token_s={(new['times'][0] - new['start']) if new.get('times') else float('nan'):.3f} "
            f"short_tps={len(new.get('times') or []) / max(1e-9, (new.get('end', 0) - (new['times'][0] if new.get('times') else 0))):.1f} "
            f"others={a.others} others_chunks_inside_per_s={inside / max(1e-9, in_span) * a.others:.1f} "
            f"others_chunks_outside_per_s={outside / max(1e-9, out_span) * a.others if out_span > 0 else float('nan'):.1f} "
            f"others_gap_max_s={gaps[-1] if gaps else float('nan'):.3f} "
            f"others_gap_p95_s={gaps[int(len(gaps) * 0.95)] if gaps else float('nan'):.3f} "
            f"errors={sum('error' in v for v in res.values())}")
    print(line)
    if a.log:
        with open(a.log, "a") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
