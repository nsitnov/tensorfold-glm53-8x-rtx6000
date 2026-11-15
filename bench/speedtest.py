#!/usr/bin/env python3
"""Speed test of an OpenAI-compatible server: tokens/s and time to first token, at 1..N parallel
requests, thinking on and off. The model name is asked from the server; the total for each level is all tokens of
the level over the wall-clock time from the first send to the last reply.

    python3 -I -B speedtest.py [--url http://127.0.0.1:8000] [--model NAME] [--levels 1,2,4,8]
                                            [--tokens 256] [--reps 1] [--thinking both|on|off] [--out FILE.json]

Levels above the server's --parallel queue (the extra requests wait their turn): keep them at or below it.

Each request is an ordinary chat request (different prose and code questions, a unique first line so that no request
reuses another's prompt cache), streamed. This client times per request: the first piece of any kind, the first word of
the answer (`content`, after the thinking), the end. Thinking and answer tokens come from `usage` (reasoning_tokens).
answer tok/s = answer tokens / (end - first answer word); thinking tok/s = thinking tokens / (first answer word -
first thinking piece). Per cell: medians over the requests, the worst first answer word, and the total = all tokens
of the cell / wall-clock time from the first send to the last reply. max_tokens counts thinking + answer.
Thinking off is sent as chat_template_kwargs.enable_thinking=false (as tools/bench_concurrent.py does); the server's
reply says which way it ran, and a request that did not run as asked is counted as an error.
Read-only for the server: it only sends requests. Other clients are not stopped, so they share the cards with the test.
"""
import argparse, json, statistics, threading, time, urllib.request, uuid

QUESTIONS = [
    "Explica en detalle qué es una cabina acústica para oficina, cuándo se usa y cuáles son sus limitaciones.",
    "Explain how a hash table handles collisions, with the trade-offs of chaining and open addressing.",
    "Write a Python function that parses a CSV file of sales and returns the total per month, with comments.",
    "Describe la historia de la ciudad de Sevilla en diez párrafos.",
    "Compare TCP and QUIC: connection set-up, head-of-line blocking, loss recovery and where each is used.",
    "Write a small REST API in Go with two endpoints (list and create items), stored in memory, with tests.",
    "¿Cuáles son las ventajas y desventajas de una bomba de calor para calentar una casa de 150 m²?",
    "Explain gradient descent and Adam to a new engineer, with a short numeric example.",
]


MODEL = "glm-5.3"
SAMPLING: dict = {}       # --temperature T: every request's sampling (default: none sent, the server's own)


def one(url, q, tokens, thinking, out, i):
    """Streamed request. Times are taken by this client: first thinking piece, first answer piece, end."""
    body = {"model": MODEL, "max_tokens": tokens, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:12]}]\n{q}"}], **SAMPLING}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); t_r = t_c = None; tf = usage = None; finish = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                d = json.loads(data)
                now = time.time()
                for ch in d.get("choices") or []:
                    de = ch.get("delta") or {}
                    if de.get("reasoning_content") and t_r is None: t_r = now
                    if de.get("content") and t_c is None: t_c = now
                    if ch.get("finish_reason"): finish = ch["finish_reason"]
                if d.get("tensorfold"): tf = d["tensorfold"]
                if d.get("usage"): usage = d["usage"]
        t_end = time.time()
        allt = usage["completion_tokens"]; rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        ran = bool(tf.get("enable_thinking")) if tf else None
        # with thinking off the server still reports enable_thinking from the template default; trust the token split
        ok = (rt > 0) == thinking or (not thinking and rt <= 1)
        ans = allt - rt
        out[i] = {"ok": ok, "tokens": allt, "think_tokens": rt, "answer_tokens": ans, "finish": finish,
                  "first_piece": min(x for x in (t_r, t_c) if x is not None) - t0 if (t_r or t_c) else None,
                  "first_answer": (t_c - t0) if t_c else None,
                  "think_tps": rt / (t_c - t_r) if (t_r and t_c and t_c > t_r and rt) else None,
                  "answer_tps": ans / (t_end - t_c) if (t_c and t_end > t_c and ans > 1) else None,
                  "tps": allt / (t_end - t0), "server_tps": tf.get("tokens_per_second") if tf else None,
                  "wall": t_end - t0, "t_end": t_end, "ran_thinking": ran}
    except Exception as e:  # noqa: BLE001 — one failed request must not stop the run
        out[i] = {"ok": False, "error": str(e)[:200], "t_end": time.time()}


def cell(url, n, tokens, thinking, rep):
    out = [None] * n
    qs = [QUESTIONS[(rep * 3 + k) % len(QUESTIONS)] for k in range(n)]
    th = [threading.Thread(target=one, args=(url, qs[k], tokens, thinking, out, k)) for k in range(n)]
    t0 = time.time()
    for x in th: x.start()
    for x in th: x.join()
    wall = max(o["t_end"] for o in out) - t0
    good = [o for o in out if o.get("ok")]
    med = lambda k: statistics.median([o[k] for o in good if o.get(k) is not None] or [float("nan")])
    r = {"streams": n, "thinking": thinking, "rep": rep, "errors": n - len(good), "wall": wall}
    if good:
        r.update(total_tps=sum(o["tokens"] for o in good) / wall, first_piece=med("first_piece"),
                 first_answer=med("first_answer"),
                 first_answer_max=max([o["first_answer"] for o in good if o.get("first_answer")] or [float("nan")]),
                 think_tokens=med("think_tokens"), answer_tokens=med("answer_tokens"),
                 think_tps=med("think_tps"), answer_tps=med("answer_tps"),
                 cut=sum(o.get("finish") == "length" for o in good))
    r["requests"] = out
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="", help="default: the first id /v1/models lists")
    ap.add_argument("--thinking", choices=("both", "on", "off"), default="both")
    ap.add_argument("--out")
    ap.add_argument("--temperature", type=float, default=None, help="every request's temperature (0: greedy); "
                    "default: none sent, the server's default sampling")
    ap.add_argument("--label", default="speedtest")
    ap.add_argument("--log", default="", help="append one parseable line a (thinking, streams) cell to this file")
    a = ap.parse_args()
    global MODEL
    if a.temperature is not None:
        SAMPLING["temperature"] = a.temperature
    with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
        MODEL = a.model or json.loads(r.read())["data"][0]["id"]
    print(f"{a.url} model {MODEL}: levels {a.levels}, {a.tokens} tokens a request, {a.reps} rep(s)", flush=True)
    levels = [int(x) for x in a.levels.split(",")]
    modes = {"both": (True, False), "on": (True,), "off": (False,)}[a.thinking]
    cells = []
    for thinking in modes:
        for n in levels:
            for rep in range(a.reps):
                c = cell(a.url, n, a.tokens, thinking, rep); cells.append(c)
                print(f"thinking={'on ' if thinking else 'off'} streams={n} rep={rep + 1}: "
                      + (f"total {c['total_tps']:.0f} tok/s, answer {c['answer_tps']:.0f} tok/s/request, "
                         f"first piece {c['first_piece']:.2f} s, first answer word {c['first_answer']:.2f} s "
                         f"(max {c['first_answer_max']:.2f}), thinking {c['think_tokens']:.0f} tok, "
                         f"answer {c['answer_tokens']:.0f} tok, cut by max_tokens {c['cut']}, errors {c['errors']}"
                         if "total_tps" in c else f"all {n} failed: {c['requests'][0].get('error')}"), flush=True)
                if a.log and "total_tps" in c:
                    with open(a.log, "a") as f:
                        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime())
                                + f"speedtest label={a.label} thinking={int(thinking)} streams={n} "
                                  f"total_tps={c['total_tps']:.1f} answer_tps={c['answer_tps']:.1f} "
                                  f"first_piece_s={c['first_piece']:.2f} errors={c['errors']}\n")
    print("\nsummary (median of the reps; times in seconds, speeds in tokens/s)")
    print(f"{'think':5} {'req':>3} {'total':>6} {'answer/req':>10} {'think/req':>9} {'1st piece':>9} "
          f"{'1st answer':>10} {'worst':>6} {'think tok':>9} {'ans tok':>7} {'cut':>4} {'err':>4}")
    for thinking in modes:
        for n in levels:
            cs = [c for c in cells if c["thinking"] == thinking and c["streams"] == n and "total_tps" in c]
            if not cs: print(f"{'on' if thinking else 'off':5} {n:>3}   no result"); continue
            m = lambda k: statistics.median(c[k] for c in cs)
            f = lambda v, w, d=0: (f"{v:>{w}.{d}f}" if v == v else f"{'-':>{w}}")
            print(f"{'on' if thinking else 'off':5} {n:>3} {f(m('total_tps'),6)} {f(m('answer_tps'),10)} "
                  f"{f(m('think_tps'),9)} {f(m('first_piece'),9,2)} {f(m('first_answer'),10,2)} "
                  f"{f(max(c['first_answer_max'] for c in cs),6,1)} {f(m('think_tokens'),9)} {f(m('answer_tokens'),7)} "
                  f"{sum(c['cut'] for c in cs):>4} {sum(c['errors'] for c in cs):>4}")
    if a.out:
        with open(a.out, "w") as f: json.dump({"args": vars(a), "t": time.time(), "cells": cells}, f, indent=1)


if __name__ == "__main__":
    main()
