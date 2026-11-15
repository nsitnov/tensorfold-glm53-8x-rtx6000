#!/usr/bin/env python3
"""N requests together against a TensorFold server, each reply's own round statistics read back (TF_GLM_ROUND_STATS):
total tokens/s, and the mean round its streams shared: rows, streams, tokens, and the milliseconds of each stage.

    python3 wide_probe.py --url http://127.0.0.1:8000 --streams 16 [--tokens 512] [--no-draft] [--temperature T]
Only the standard library; requests go to --url alone. One parseable line (also appended to --log).
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPTS = [
    "Write a long, detailed technical explanation of how a B-tree index works in a database, with examples.",
    "Explain step by step how TCP congestion control works, covering slow start, congestion avoidance and fast recovery.",
    "Write a detailed guide to implementing a hash map in C, including resizing and collision handling, with code.",
    "Describe the history of the Roman Republic from its founding to Augustus in a long, structured essay.",
    "Write a thorough tutorial on Python generators and coroutines with many code examples.",
    "Explain how a modern CPU executes instructions: pipelining, branch prediction, caches, out-of-order execution.",
    "Write a long review of the main sorting algorithms, their complexity and when to use each, with pseudocode.",
    "Describe in detail how the Linux kernel schedules processes and threads, including CFS and real-time classes.",
]


THINKING: bool | None = False


def ask(url: str, model: str, prompt: str, tokens: int, draft: bool, sampling: dict, timeout: float, out: list,
        stream: bool = True, ignore_eos: bool = False, logprobs: int | None = None) -> None:
    body = {"model": model, "max_tokens": tokens, **sampling, "stream": stream,
            "messages": [{"role": "user", "content": prompt}]}
    if THINKING is not None:               # (--thinking default: nothing sent, the server's own way)
        body["chat_template_kwargs"] = {"enable_thinking": THINKING}
    if ignore_eos:
        body["ignore_eos"] = True
    if logprobs is not None:                 # --logprobs-streams: this stream asks logprobs (top_logprobs N)
        body["logprobs"], body["top_logprobs"] = True, logprobs
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = stats = usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not stream:
                d = json.load(resp)
                if "error" in d:
                    raise RuntimeError(json.dumps(d["error"])[:200])
                stats, usage = d.get("tensorfold"), d.get("usage")
            for raw in resp if stream else ():
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                d = json.loads(line[5:])
                if "error" in d:
                    raise RuntimeError(json.dumps(d["error"])[:200])
                for ch in d.get("choices") or []:
                    de = ch.get("delta") or {}
                    if first is None and (de.get("content") or de.get("reasoning_content")):
                        first = time.time()
                stats, usage = d.get("tensorfold") or stats, d.get("usage") or usage
    except Exception as exc:  # noqa: BLE001
        out.append({"error": str(exc)[:200]})
        return
    end = time.time()
    out.append({"tokens": int((usage or {}).get("completion_tokens") or 0), "t0": t0, "first": first or end,
                "end": end, "stats": stats or {}})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--streams", type=int, default=8)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--no-draft", action="store_true")
    p.add_argument("--no-stream", action="store_true", help="whole replies instead of streamed ones (the HTTP side's "
                   "share of the host time between rounds)")
    p.add_argument("--ignore-eos", action="store_true", help="replies run to --tokens (a fake checkpoint's replies "
                   "end at random otherwise)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompts-file", default="", help="one prompt a line instead of the built-in eight (sent as they "
                   "are, no variant suffix): another benchmark's own prompts, for a like-for-like run")
    p.add_argument("--thinking", choices=("off", "on", "default"), default="off")
    p.add_argument("--logprobs-streams", type=int, default=0, help="the first K streams ask logprobs (their cost)")
    p.add_argument("--top-logprobs", type=int, default=5)
    p.add_argument("--label", default="wide")
    p.add_argument("--log", default="")
    p.add_argument("--timeout", type=float, default=180.0)
    a = p.parse_args()
    global THINKING
    THINKING = {"off": False, "on": True, "default": None}[a.thinking]
    prompts = [PROMPTS[i % len(PROMPTS)] + f" (variant {i})" for i in range(a.streams)]
    if a.prompts_file:
        own = [x for x in open(a.prompts_file, encoding="utf-8").read().splitlines() if x.strip()]
        prompts = [own[i % len(own)] for i in range(a.streams)]
    with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    sampling = {"temperature": a.temperature} if a.temperature <= 0 else {"temperature": a.temperature, "top_p": 0.95}
    out: list = []
    threads = [threading.Thread(target=ask, args=(a.url, model, prompts[i],
                                                  a.tokens, not a.no_draft, sampling, a.timeout, out,
                                                  not a.no_stream, a.ignore_eos,
                                                  a.top_logprobs if i < a.logprobs_streams else None))
               for i in range(a.streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = [o for o in out if "error" not in o]
    errors = len(out) - len(ok)
    if not ok:
        line = f"wide_probe label={a.label} streams={a.streams} result=FAIL errors={errors} first=\"{out[0].get('error') if out else ''}\""
    else:
        wall = max(o["end"] for o in ok) - min(o["t0"] for o in ok)
        gen = max(o["end"] for o in ok) - min(o["first"] for o in ok)
        total = sum(o["tokens"] for o in ok)
        rm = [o["stats"].get("round_ms") or {} for o in ok]

        def mean(key: str) -> float:
            vals = [float(r[key]) for r in rm if key in r]
            return statistics.fmean(vals) if vals else float("nan")

        stages = {k: mean(k) for k in ("plan", "forward", "sample", "keep", "draft", "emit", "between")}
        rnd = sum(v for v in stages.values() if v == v)
        rounds = mean("rounds")
        per_round = sum(o["tokens"] / max(1, int((o["stats"].get("round_ms") or {}).get("rounds") or o["stats"].get("rounds") or 1))
                        for o in ok)
        drafted = sum(int(o["stats"].get("drafted") or 0) for o in ok)
        accepted = sum(int(o["stats"].get("accepted") or 0) for o in ok)
        line = (f"wide_probe label={a.label} streams={a.streams} draft={int(not a.no_draft)} "
                f"{f'logprobs_streams={a.logprobs_streams} ' if a.logprobs_streams else ''}tokens={total} "
                f"total_tps={total / wall:.1f} gen_tps={total / gen:.1f} first_token_s={statistics.median(o['first'] - o['t0'] for o in ok):.3f} "
                f"rounds={rounds:.0f} rows_a_round={mean('rows'):.1f} streams_a_round={mean('streams'):.1f} "
                f"tokens_a_round={per_round:.1f} kept={accepted / drafted if drafted else float('nan'):.3f} "
                f"round_ms={rnd:.2f} forward_ms={stages['forward']:.2f} draft_ms={stages['draft']:.2f} "
                f"sample_ms={stages['sample']:.2f} between_ms={stages['between']:.2f} plan_ms={stages['plan']:.2f} "
                f"keep_ms={stages['keep']:.2f} emit_ms={stages['emit']:.2f} errors={errors}")
        # TF_GLM_BETWEEN_STATS: "between" by part (b_sched, b_send, b_agree_gather ...), when the server reports them
        parts = sorted({k for r in rm for k in r if k.startswith("b_")})
        line += "".join(f" {k}_ms={mean(k):.3f}" for k in parts)
        line += f" streamed={int(not a.no_stream)}"
        counters = [r.get("graphs") or {} for r in rm]
        if any(counters):                  # the server's graph counters as the last reply saw them (rounds replayed
            #                                from a graph / run eagerly, graphs captured, graphs that gave way)
            last = max(counters, key=lambda c: c.get("graph", 0) + c.get("eager", 0))
            line += "".join(f" {k}={last[k]}" for k in ("graph", "eager", "captured", "evicted", "mtp_graph") if k in last)
        ds = [r["draft_stats"] for r in rm if r.get("draft_stats")]
        if ds:                             # TF_GLM_DRAFT_STATS: the server's verified drafts by chain probability
            #                                (tenths 0.0-0.1 ... 0.9-1.0) as kept/verified, since the server started
            last = max(ds, key=lambda t: sum(c[0] for c in t))
            line += " draft_stats=" + ",".join(f"{c[1]}/{c[0]}" for c in last)
    print(line)
    if a.log:
        with open(a.log, "a") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + line + "\n")
    return 0 if ok and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
