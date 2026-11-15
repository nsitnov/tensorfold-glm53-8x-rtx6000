#!/usr/bin/env python3
"""Concurrent == solo on a running server, at the width that is served: N different prompts one at a time (greedy,
thinking off, replies of exactly --tokens tokens), then all N at once; every reply together must be the reply alone.

    together_check.py --url http://127.0.0.1:8000 --prompts-file FILE [--streams 32] [--tokens 256] --label L [--log F]

One line:  together_check label=L streams=32 tokens=256 same=32/32 first_differs_at=- result=PASS
(first_differs_at: for the first reply that differs, its index and the character where it leaves the solo reply.)
Standard library only; requests go to --url alone.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request


def ask(url: str, model: str, prompt: str, tokens: int, timeout: float) -> str:
    body = {"model": model, "max_tokens": tokens, "temperature": 0, "stream": False, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"].get("content") or ""
    except Exception as exc:  # noqa: BLE001
        return f"\0ERROR {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--streams", type=int, default=32)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--label", default="together")
    p.add_argument("--log", default="")
    p.add_argument("--timeout", type=float, default=300.0)
    a = p.parse_args()
    own = [x for x in open(a.prompts_file, encoding="utf-8").read().splitlines() if x.strip()]
    prompts = [own[i % len(own)] + ("" if i < len(own) else f" ({i})") for i in range(a.streams)]
    with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    solo = [ask(a.url, model, q, a.tokens, a.timeout) for q in prompts]
    got = [""] * len(prompts)

    def one(i: int) -> None:
        got[i] = ask(a.url, model, prompts[i], a.tokens, a.timeout)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    errors = sum(x.startswith("\0ERROR") for x in solo + got)
    same = sum(x == y for x, y in zip(solo, got))
    where = "-"
    for i, (x, y) in enumerate(zip(solo, got)):
        if x != y:
            where = f"{i}:{next((k for k, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))}"
            break
    line = (f"together_check label={a.label} streams={a.streams} tokens={a.tokens} same={same}/{len(prompts)} "
            f"first_differs_at={where} errors={errors} "
            f"result={'PASS' if same == len(prompts) and not errors else 'FAIL'}")
    print(line)
    if a.log:
        with open(a.log, "a") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + line + "\n")
    return 0 if same == len(prompts) and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
