#!/usr/bin/env python3
"""Window GROW's probes (TF_GLM_REPLY_GROW, TF_GLM_CALL_STREAM) against a test server. Standard library only; requests
go to --url alone.

  grow_probe.py long --url U --long 200000 --label L [--log F] [--cut 24000] [--shorts 6] [--every 12]
      A ~200k-token prompt whose max_tokens fills the context (as the API gateway sizes an agent's request), greedy,
      streamed, its reply cut by the client after --cut characters: first alone, then again with --shorts short chat
      requests arriving (the first while its prompt is read, the others every --every s while it decodes). Each short
      request's first token must come within --short-limit s (the production failure: 145 s), and the long reply's
      text must be its text alone (the shorter a prefix of the longer, at least --min-chars of it). One line:
        grow_probe label=L mode=long prompt_tokens=.. max_tokens=.. solo_chars=.. together_chars=.. same=1
          solo_first_token_s=.. together_first_token_s=.. short_ttft_s=a,b,.. short_ttft_max_s=.. short_errors=0
          growth="{..}" result=PASS|FAIL

  grow_probe.py tool --url U --label L [--log F] [--lines 150] [--choice required|auto] [--thinking]
      A forced tool call that writes a file of --lines lines, greedy, thinking off: streamed (chat), then the same
      request whole, then parallel_tool_calls false streamed, then /v1/responses streamed. The streamed argument
      fragments must arrive over time (many events, the first long before the last) and make exactly the whole
      reply's arguments. One line:
        grow_probe label=L mode=tool fragments=.. span_s=.. max_gap_s=.. same=1 single_fragments=.. single_same=1
          responses_deltas=.. responses_same=1 result=PASS|FAIL
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


def _post(url, path, body, timeout=3600):
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _events(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            yield json.loads(line[5:])


def _emit(line, log):
    print(line, flush=True)
    if log:
        with open(log, "a") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n")


# -- long ---------------------------------------------------------------------------------------------------------------

def _long_body(model, doc, max_tokens):
    return {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": doc}]}


def _long_run(url, body, cut, out):
    """The long request streamed until ``cut`` characters of text arrived (then the client leaves)."""
    t0 = time.perf_counter()
    text, first = [], None
    try:
        with _post(url, "/v1/chat/completions", body) as r:
            for d in _events(r):
                if "error" in d:
                    raise RuntimeError(json.dumps(d["error"])[:300])
                for c in d.get("choices") or []:
                    piece = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                    if piece:
                        first = first or time.perf_counter()
                        text.append(piece)
                if sum(map(len, text)) >= cut:
                    break
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:300]
    out.update(text="".join(text), first=(first or time.perf_counter()) - t0, end=time.perf_counter() - t0, t0=t0)


def _short(url, model, i, out):
    body = {"model": model, "max_tokens": 64, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"In two sentences: what is a {['river', 'star', 'cell', 'bridge', 'storm', 'seed', 'lens', 'tide'][i % 8]}?"}]}
    t0, first, stats = time.perf_counter(), None, None
    try:
        with _post(url, "/v1/chat/completions", body, timeout=900) as r:
            for d in _events(r):
                stats = d.get("tensorfold") or stats
                for c in d.get("choices") or []:
                    if (c.get("delta") or {}).get("content") and first is None:
                        first = time.perf_counter()
        out[i] = {"ttft": (first or time.perf_counter()) - t0, "stats": stats or {}}
    except Exception as exc:  # noqa: BLE001
        out[i] = {"error": str(exc)[:200]}


def long_mode(a, model):
    lines = max(1, round((a.long - OVERHEAD) / PER_LINE))
    doc = document(lines, 4242) + "\n\nRepeat every record line above, one per line, in order, from Record 0 on."
    # its prompt's tokens (and its rows kept: the runs below resume them, as an agent's next step does)
    with _post(a.url, "/v1/chat/completions", {**_long_body(model, doc, 1), "stream": False}) as r:
        prompt_tokens = json.load(r)["usage"]["prompt_tokens"]
    # the largest max_tokens that fits beside it, as the API gateway sizes an agent's request (/health: the window)
    with urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=30) as r:
        context = int(json.load(r).get("context_length") or a.context)
    max_tokens = context - prompt_tokens
    body = _long_body(model, doc, max_tokens)
    solo: dict = {}
    _long_run(a.url, body, a.cut, solo)
    time.sleep(2)
    together: dict = {}
    th = threading.Thread(target=_long_run, args=(a.url, body, a.cut, together))
    th.start()
    shorts: dict = {}
    waits = [a.first_short] + [a.every] * (a.shorts - 1)
    sth = []
    for i, w in enumerate(waits):
        time.sleep(w)
        t = threading.Thread(target=_short, args=(a.url, model, i, shorts))
        t.start()
        sth.append(t)
    for t in sth:
        t.join()
    th.join()
    s1, s2 = solo.get("text", ""), together.get("text", "")
    n = min(len(s1), len(s2))
    same = int(n >= a.min_chars and s1[:n] == s2[:n])
    ttfts = [shorts[i]["ttft"] for i in sorted(shorts) if "ttft" in shorts[i]]
    errors = sum("error" in v for v in shorts.values()) + int("error" in solo) + int("error" in together)
    last = max((shorts[i] for i in shorts if "stats" in shorts[i]), key=lambda v: v["ttft"], default={})
    rm = (last.get("stats") or {}).get("round_ms") or {}
    ok = same and ttfts and max(ttfts) <= a.short_limit and len(ttfts) == a.shorts and not errors
    return (f"grow_probe label={a.label} mode=long prompt_tokens={prompt_tokens} max_tokens={max_tokens} "
            f"solo_chars={len(s1)} together_chars={len(s2)} same={same} "
            f"solo_first_token_s={solo.get('first', float('nan')):.2f} "
            f"together_first_token_s={together.get('first', float('nan')):.2f} "
            f"short_ttft_s={','.join(f'{x:.2f}' for x in ttfts)} "
            f"short_ttft_max_s={max(ttfts) if ttfts else float('nan'):.2f} short_errors={errors} "
            f"growth=\"{json.dumps(rm.get('reply_grow'), separators=(',', ':'))}\" "
            f"extents=\"{json.dumps(rm.get('extents'), separators=(',', ':'))}\" "
            f"errors=\"{(solo.get('error') or together.get('error') or '')[:120]}\" result={'PASS' if ok else 'FAIL'}")


# -- tool ---------------------------------------------------------------------------------------------------------------

TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a text file.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string", "description": "the file's path"},
                                     "content": {"type": "string", "description": "the whole text of the file"},
                                     "overwrite": {"type": "boolean"}},
    "required": ["path", "content"]}}}]


def _tool_body(model, lines, stream, choice="required", thinking=False, **extra):
    return {"model": model, "max_tokens": 16000, "temperature": 0, "stream": stream, "tools": TOOLS,
            "tool_choice": choice, "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": f"Use write_file to create notes/lines.txt with {lines} numbered "
                                                     "lines, each a different short sentence about the sea. "
                                                     "Set overwrite to true."}], **extra}


def _chat_stream(url, body):
    t0, names, args, times, events = time.perf_counter(), {}, {}, [], []
    with _post(url, "/v1/chat/completions", body) as r:
        for d in _events(r):
            events.append(time.perf_counter() - t0)
            for c in d.get("choices") or []:
                for tc in (c.get("delta") or {}).get("tool_calls") or []:
                    i = tc["index"]
                    names[i] = names.get(i, "") + (tc.get("function") or {}).get("name", "")
                    frag = (tc.get("function") or {}).get("arguments") or ""
                    args[i] = args.get(i, "") + frag
                    if frag:
                        times.append(time.perf_counter() - t0)
    gaps = [b - x for x, b in zip(events, events[1:])]
    return [(names[i], args[i]) for i in sorted(names)], times, max(gaps, default=0.0)


def tool_mode(a, model):
    calls, times, gap = _chat_stream(a.url, _tool_body(model, a.lines, True, a.choice, a.thinking))
    with _post(a.url, "/v1/chat/completions", _tool_body(model, a.lines, False, a.choice, a.thinking)) as r:
        whole = json.load(r)["choices"][0]["message"].get("tool_calls") or []
    want = [(c["function"]["name"], c["function"]["arguments"]) for c in whole]
    same = int(bool(want) and calls == want)
    single, stimes, _ = _chat_stream(a.url, _tool_body(model, a.lines, True, a.choice, a.thinking,
                                                               parallel_tool_calls=False))
    single_same = int(bool(want) and single == want[:1])
    # /v1/responses, streamed: its function_call_arguments.delta events
    rbody = {"model": model, "input": _tool_body(model, a.lines, True)["messages"][0]["content"], "stream": True,
             "max_output_tokens": 16000, "temperature": 0, "tool_choice": a.choice,
             # (as the chat requests: without it the served default, thinking, applied to this one alone - window
             # GROW's tool-auto line had another reply there, no call)
             "chat_template_kwargs": {"enable_thinking": a.thinking},
             "tools": [{"type": "function", **t["function"]} for t in TOOLS]}
    deltas, done, items, status = [], [], [], ""
    try:
        with _post(a.url, "/v1/responses", rbody) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:])
                if ev.get("type") == "response.function_call_arguments.delta":
                    deltas.append(ev["delta"])
                elif ev.get("type") == "response.function_call_arguments.done":
                    done.append(ev["arguments"])
                elif ev.get("type") == "response.output_item.done":
                    it = ev.get("item") or {}
                    text = "".join(c.get("text", "") for c in it.get("content") or [] if isinstance(c, dict))
                    items.append(f"{it.get('type')}:{len(text) or len(it.get('arguments') or '')}")
                elif ev.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                    status = ev["type"].split(".")[1]
    except Exception as exc:  # noqa: BLE001
        done = [f"error {exc}"]
    rsame = int(bool(done) and "".join(deltas) == done[0] == (want[0][1] if want else None))
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    ok = same and single_same and rsame and len(times) >= 20 and span >= 2.0 and len(stimes) >= 20 and len(deltas) >= 20
    if a.thinking and not want and not times and not deltas and not single:
        ok = True                                        # (thinking: the model may answer in text; then nowhere a call)
    return (f"grow_probe label={a.label} mode=tool calls={len(want)} args_chars={len(want[0][1]) if want else 0} "
            f"fragments={len(times)} first_fragment_s={times[0] if times else float('nan'):.2f} span_s={span:.2f} "
            f"max_gap_s={gap:.2f} same={same} single_fragments={len(stimes)} single_same={single_same} "
            f"responses_deltas={len(deltas)} responses_same={rsame} responses_items={','.join(items) or '-'} "
            f"responses_status={status or '-'} thinking={int(a.thinking)} result={'PASS' if ok else 'FAIL'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["long", "tool"])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--label", default="grow")
    p.add_argument("--log", default="")
    p.add_argument("--long", type=int, default=200000)
    p.add_argument("--context", type=int, default=262144)
    p.add_argument("--cut", type=int, default=24000, help="characters of the long reply read before the client leaves")
    p.add_argument("--min-chars", type=int, default=12000)
    p.add_argument("--shorts", type=int, default=6)
    p.add_argument("--first-short", type=float, default=5.0, help="s after the long request: while its prompt is read")
    p.add_argument("--every", type=float, default=12.0)
    p.add_argument("--short-limit", type=float, default=10.0)
    p.add_argument("--lines", type=int, default=150)
    p.add_argument("--choice", default="required", help="the tool call's tool_choice (required | auto)")
    p.add_argument("--thinking", action="store_true", help="thinking on for every request (the served default)")
    a = p.parse_args()
    with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    line = long_mode(a, model) if a.mode == "long" else tool_mode(a, model)
    _emit(line, a.log)
    return 0 if "result=PASS" in line else 1


if __name__ == "__main__":
    raise SystemExit(main())
