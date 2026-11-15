#!/usr/bin/env python3
"""Single-request decode speed of a running server, and where its rounds' time goes.

    python3 speed_probe.py [--url http://127.0.0.1:$PORT] [--tokens 384] [--label NAME] [--out FILE.json]
                                        [--kernel-forward-ms X] [--log bench.log]

Fixed prompts (prose, code, non-English), one request at a time, thinking off, temperature 0, streamed (the first
token's time is the client's clock). Each prompt runs twice: serial ("draft": false: one token a round, so its
tokens/s is 1 / round time) and drafted (the server's MTP drafts). From each reply's stats: rounds, drafts verified
and kept, and with TF_GLM_ROUND_STATS=1 the mean milliseconds a round spent in each stage (forward to its sync,
sample, draft, between ...). The drafted reply must equal the serial one (the engine's guarantee), token for token.

The last line is for programs (appended to --log when given):
  speed_probe label=L serial_tps=.. drafted_tps=.. speedup=.. tokens_per_round=.. kept=.. first_token_s=..
              serial_round_ms=.. serial_forward_ms=.. drafted_round_ms=.. drafted_forward_ms=.. draft_ms=.. sample_ms=..
              between_ms=.. comm_ms=.. equal=1 errors=0
tps are tokens/s over the decode (the reply's tokens after the first, over the time from the first token to the end),
medians over the prompts. comm_ms: with --kernel-forward-ms (a round's forward on one rank without communication:
decode_profile.py on an idle GPU), the serial forward's milliseconds the transport adds.
Only the standard library; requests go to --url alone.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

PROMPTS = [
    "Explain how a hash table handles collisions, with the trade-offs of chaining and open addressing.",
    "Write a Python function that parses a CSV file of sales and returns the total per month, with comments.",
    "Describe la historia de la ciudad de Sevilla en diez párrafos.",
]


POLICY = ""          # --policy N: "tf_policy" of the drafted requests (the single-stream engine's drafts a round)
SAMPLING: dict = {"temperature": 0}      # --temperature T [--top-p P --top-k K --seed S]: every request's sampling


def ask(url: str, model: str, prompt: str, tokens: int, draft: bool, timeout: float) -> dict:
    body = {"model": model, "max_tokens": tokens, **SAMPLING, "stream": True, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    if not draft:
        body["draft"] = False
    elif POLICY:
        body["tf_policy"] = POLICY
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = stats = usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            if "error" in d:
                raise RuntimeError(json.dumps(d["error"])[:300])
            for ch in d.get("choices") or []:
                de = ch.get("delta") or {}
                if first is None and (de.get("content") or de.get("reasoning_content")):
                    first = time.time()
            stats, usage = d.get("tensorfold") or stats, d.get("usage") or usage
    end = time.time()
    n = int((usage or {}).get("completion_tokens") or 0)
    stats = stats or {}
    rounds = int(stats.get("rounds") or 0)
    return {"tokens": n, "first_token_s": (first or end) - t0, "wall_s": end - t0,
            "tps": (n - 1) / (end - first) if first and n > 1 and end > first else float("nan"),
            "server_tps": (n - 1) / stats["decode_s"] if stats.get("decode_s") else float("nan"),
            "rounds": rounds, "drafted": int(stats.get("drafted") or 0), "accepted": int(stats.get("accepted") or 0),
            "round_ms": stats.get("round_ms") or {}, "ids": stats.get("token_ids"),
            "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0), "prefill_s": stats.get("prefill_s")}


def med(values) -> float:
    values = [v for v in values if v is not None and v == v]
    return statistics.median(values) if values else float("nan")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    port = os.environ.get("PORT") or "8000"
    p.add_argument("--url", default=f"http://127.0.0.1:{port}")
    p.add_argument("--model", default="")
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--label", default="probe")
    p.add_argument("--prompts", type=int, default=len(PROMPTS))
    p.add_argument("--kernel-forward-ms", type=float, default=0.0)
    p.add_argument("--policy", default="", help="drafts a round for the drafted requests (\"tf_policy\": a server "
                   "without --parallel takes it per request, up to 5 with its captured graphs)")
    p.add_argument("--out", default="")
    p.add_argument("--log", default="")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--temperature", type=float, default=0.0, help="0: greedy (default); the server's own default "
                   "for real requests is 1.0 with top_p 0.95")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234, help="with --temperature > 0: the keyed sampler's seed (the "
                   "serial and the drafted run draw the same tokens: the replies must still be equal)")
    a = p.parse_args(argv)
    global POLICY, SAMPLING
    POLICY = a.policy
    if a.temperature > 0:
        SAMPLING = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "seed": a.seed}
    with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = a.model or json.loads(r.read())["data"][0]["id"]
    rows, errors = [], 0
    ask(a.url, model, "Say hello.", 8, True, a.timeout)                     # the first request's one-time costs
    for prompt in PROMPTS[:a.prompts]:
        try:
            serial = ask(a.url, model, prompt, a.tokens, False, a.timeout)
            drafted = ask(a.url, model, prompt, a.tokens, True, a.timeout)
        except Exception as exc:                                            # noqa: BLE001
            errors += 1
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        equal = serial["ids"] is not None and serial["ids"] == drafted["ids"]
        rows.append((serial, drafted, equal))
        kept = drafted["accepted"] / drafted["drafted"] if drafted["drafted"] else float("nan")
        print(f"  prompt {len(rows)} ({serial['prompt_tokens']} tokens): serial {serial['tps']:.1f} tok/s "
              f"({serial['tokens']} tok, {serial['rounds']} rounds), drafted {drafted['tps']:.1f} tok/s "
              f"({drafted['tokens']} tok, {drafted['rounds']} rounds, {drafted['accepted']} of {drafted['drafted']} "
              f"drafts kept = {kept:.0%}), first token {drafted['first_token_s']:.2f} s, "
              f"{'equal' if equal else 'DIFFERENT REPLIES'}", flush=True)
        for name, r in (("serial", serial), ("drafted", drafted)):
            if r["round_ms"]:
                print(f"      {name} round (ms): " + ", ".join(f"{k} {v}" for k, v in r["round_ms"].items()
                                                                 if not isinstance(v, dict)), flush=True)
    if not rows:
        line = f"speed_probe label={a.label} result=FAIL errors={errors}"
    else:
        s, d = [r[0] for r in rows], [r[1] for r in rows]

        def stage(runs, key):
            return med([r["round_ms"].get(key) for r in runs])

        def whole(runs):
            return med([1e3 * (r["tokens"] - 1) / r["tps"] / max(1, r["rounds"]) if r["tps"] == r["tps"] else None
                        for r in runs])

        kept = sum(r["accepted"] for r in d) / max(1, sum(r["drafted"] for r in d))
        tpr = sum(r["tokens"] for r in d) / max(1, sum(r["rounds"] for r in d))
        comm = stage(s, "forward") - a.kernel_forward_ms if a.kernel_forward_ms else float("nan")
        line = (f"speed_probe label={a.label} serial_tps={med([r['tps'] for r in s]):.1f} "
                f"drafted_tps={med([r['tps'] for r in d]):.1f} "
                f"speedup={med([r['tps'] for r in d]) / med([r['tps'] for r in s]):.2f} tokens_per_round={tpr:.2f} "
                f"kept={kept:.3f} first_token_s={med([r['first_token_s'] for r in d]):.3f} "
                f"serial_round_ms={whole(s):.2f} serial_forward_ms={stage(s, 'forward'):.2f} "
                f"drafted_round_ms={whole(d):.2f} drafted_forward_ms={stage(d, 'forward'):.2f} "
                f"draft_ms={stage(d, 'draft'):.2f} sample_ms={stage(d, 'sample'):.2f} "
                f"between_ms={stage(d, 'between'):.2f} comm_ms={comm:.2f} "
                f"equal={int(all(r[2] for r in rows))} errors={errors}")
    print(line, flush=True)
    if a.log:
        with open(a.log, "a") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"label": a.label, "url": a.url, "rows": [{"serial": r[0], "drafted": r[1], "equal": r[2]}
                                                                 for r in rows]}, f)
    return 0 if rows and not errors and all(r[2] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
