#!/usr/bin/env python3
"""Requests with a long prompt against a local OpenAI-style server: the time to the first token (the prompt's read)
and the tokens/s of the decode after it; optionally a second request that extends the first as an agent's next step
does (the first prompt, its reply, a new question: what a kept prompt saves), or several such requests together.

    longctx.py --url http://127.0.0.1:8000 --tokens 16000 [--max-tokens 256] --label NAME [--log FILE]
               [--extend] [--streams N] [--seed S]
    longctx.py URL LINES MAX_TOKENS LABEL [LOG]            (the first form of this tool: LINES records of ~17 tokens)

One line a request (also appended to --log with a time stamp), the keys the results ledger reads first:
  longctx label=L prompt_tokens=.. cached_tokens=.. completion_tokens=.. first_token_s=.. prefill_tps=.. decode_s=..
          decode_tps=.. total_s=..   then, when the server reports them (TF_GLM_ROUND_STATS): rounds, rows a round,
          forward / draft / between ms a round, and its graph counters (graph, eager, captured, evicted)
--extend adds a line L-ext for the second request; --streams N sends N different prompts at once and prints ONE line
(label L): tokens summed, first_token_s the median, decode_tps the total over the time from the first token of any to
the end of all, streams=N.  Standard library only; requests go to --url alone.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
QUESTION = ("\n\nWrite a long, plain description of how such station records could be summarised, in about 300 "
            "words.")
FOLLOW = "Now list, in about 200 words, three checks you would run on these records before trusting the summary."
PER_LINE, OVERHEAD = 17.3, 40          # tokens a record line and around the document (measured: 128 lines = 2,238)


def document(lines: int, seed: int = 0) -> str:
    return "\n".join(f"Record {i}: station {WORDS[(i + seed) % 16]}-{WORDS[(i * 7 + seed) % 16]} reported value "
                     f"{(i + 11 * seed) * 37 % 1000} at step {i}." for i in range(lines))


def ask(url: str, model: str, messages: list, max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = usage = stats = None
    text: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                d = json.loads(line[5:])
                if "error" in d:
                    raise RuntimeError(json.dumps(d["error"])[:300])
                usage = d.get("usage") or usage
                stats = d.get("tensorfold") or d.get("stats") or d.get("x_stats") or stats
                for c in d.get("choices") or []:
                    delta = c.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        text.append(piece)
                        if first is None:
                            first = time.time()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:300], "t0": t0}
    t1 = time.time()
    usage = usage or {}
    return {"t0": t0, "first": first or t1, "end": t1, "text": "".join(text), "stats": stats or {},
            "prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def extras(stats: dict) -> str:
    rm = stats.get("round_ms") or {}
    g = rm.get("graphs") or {}
    out = []
    for key, name in (("rounds", "rounds"), ("rows", "rows_a_round"), ("forward", "forward_ms"), ("draft", "draft_ms"),
                      ("between", "between_ms")):
        if key in rm:
            out.append(f"{name}={rm[key]}")
    for key in ("graph", "eager", "captured", "evicted", "mtp_graph"):
        if key in g:
            out.append(f"{key}={g[key]}")
    if "cached" in stats:
        out.append(f"server_cached={stats['cached']}")
    fills = stats.get("fill_ms")                      # TF_GLM_FILL_STATS: the prompt's chunks [first row, rows, ms]
    if fills:
        whole = [f for f in fills if f[1] == max(x[1] for x in fills)]
        out.append(f"chunks={len(fills)} chunk_rows={whole[0][1]} chunk_ms_first={whole[0][2]} "
                   f"chunk_ms_last={whole[-1][2]} chunk_ms=" + ",".join(f"{f[0]}:{f[2]:.0f}" for f in fills))
    return (" " + " ".join(out)) if out else ""


def line_of(label: str, got: dict) -> str:
    if "error" in got:
        return f"longctx label={label} result=FAIL error=\"{got['error'][:200]}\""
    pt, ct, cached = got["prompt"], got["completion"], got["cached"]
    ttft, dec = got["first"] - got["t0"], got["end"] - got["first"]
    return (f"longctx label={label} prompt_tokens={pt} cached_tokens={cached} completion_tokens={ct} "
            f"first_token_s={ttft:.3f} "
            f"prefill_tps={(pt - (cached or 0)) / ttft if pt and ttft > 0 else float('nan'):.0f} decode_s={dec:.2f} "
            f"decode_tps={(ct - 1) / dec if ct and dec > 0 else float('nan'):.1f} total_s={got['end'] - got['t0']:.2f}"
            + extras(got["stats"]))


def emit(line: str, log: str | None) -> None:
    print(line, flush=True)
    if log:
        with open(log, "a") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n")


def main() -> int:
    if len(sys.argv) >= 5 and not sys.argv[1].startswith("-"):          # the first form: URL LINES MAX_TOKENS LABEL [LOG]
        argv = ["--url", sys.argv[1], "--lines", sys.argv[2], "--max-tokens", sys.argv[3], "--label", sys.argv[4]]
        argv += ["--log", sys.argv[5]] if len(sys.argv) > 5 else []
    else:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--tokens", type=int, default=0, help="about this many prompt tokens")
    p.add_argument("--lines", type=int, default=0, help="or: this many record lines")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--label", default="longctx")
    p.add_argument("--log", default="")
    p.add_argument("--extend", action="store_true", help="then a second request: the first, its reply, a new question")
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=900.0)
    a = p.parse_args(argv)
    lines = a.lines or max(1, round((a.tokens - OVERHEAD) / PER_LINE))
    with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    if a.streams > 1:
        got: list = [None] * a.streams

        def one(i: int) -> None:
            got[i] = ask(a.url, model, [{"role": "user", "content": document(lines, a.seed + 1 + i) + QUESTION}],
                         a.max_tokens, a.timeout)

        threads = [threading.Thread(target=one, args=(i,)) for i in range(a.streams)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok = [g for g in got if g and "error" not in g]
        if len(ok) < a.streams:
            bad = next(g for g in got if not g or "error" in g) or {}
            emit(f"longctx label={a.label} result=FAIL streams={a.streams} errors={a.streams - len(ok)} "
                 f"error=\"{bad.get('error', '')[:160]}\"", a.log)
            return 1
        t0, first_any, end = min(g["t0"] for g in ok), min(g["first"] for g in ok), max(g["end"] for g in ok)
        last_first = max(g["first"] for g in ok)
        pt, ct = sum(g["prompt"] or 0 for g in ok), sum(g["completion"] or 0 for g in ok)
        cached = sum(g["cached"] or 0 for g in ok)
        dec = end - first_any
        eager = [((g["stats"].get("round_ms") or {}).get("graphs") or {}).get("eager") for g in ok]
        emit(f"longctx label={a.label} prompt_tokens={pt} cached_tokens={cached} completion_tokens={ct} "
             f"first_token_s={statistics.median(g['first'] - g['t0'] for g in ok):.2f} "
             f"prefill_tps={(pt - cached) / (last_first - t0) if last_first > t0 else float('nan'):.0f} "
             f"decode_s={dec:.2f} decode_tps={(ct - len(ok)) / dec if dec > 0 else float('nan'):.1f} "
             f"total_s={end - t0:.2f} streams={a.streams} last_first_token_s={last_first - t0:.2f}"
             + extras(max(ok, key=lambda g: g['end'])["stats"]), a.log)
        return 0
    messages = [{"role": "user", "content": document(lines, a.seed) + QUESTION}]
    got1 = ask(a.url, model, messages, a.max_tokens, a.timeout)
    emit(line_of(a.label, got1), a.log)
    if "error" in got1:
        return 1
    if a.extend:
        messages += [{"role": "assistant", "content": got1["text"]}, {"role": "user", "content": FOLLOW}]
        got2 = ask(a.url, model, messages, a.max_tokens, a.timeout)
        emit(line_of(a.label + "-ext", got2), a.log)
        if "error" in got2:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
