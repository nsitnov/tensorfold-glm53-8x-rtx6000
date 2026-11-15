#!/usr/bin/env python3
"""A growing agent conversation against a local OpenAI-style server: what a kept prompt saves (TF_GLM_KEEP_SLOTS),
and whether a resumed step replies as the same step read fresh.

    reuse_check.py --url http://127.0.0.1:8000 --tokens 16000 --grow 2000 --steps 3 --label NAME [--log FILE]
                   [--save FILE | --compare FILE] [--beside N] [--max-tokens 128] [--seed S]

Step 1 sends a document of about --tokens tokens and a question; every later step sends the conversation so far (the
model's own replies, greedy) and a "tool result" of about --grow tokens. One standard ``longctx`` line a step (label
NAME-s1, NAME-s2, ...: prompt_tokens, cached_tokens, first_token_s, prefill_tps of the tokens actually read,
decode_tps), which the results ledger reads, then one summary line:

  reuse_check label=NAME steps=3 cached=0,16384,16384 first_token_s=4.41,0.71,0.80 same=1 result=PASS

--save FILE writes each step's prompt size and reply; --compare FILE checks them against a saved run (``same``): run
it once on a load without TF_GLM_KEEP_SLOTS (every step read fresh) with --save and once with it and --compare - the
replies must be equal, text and token count (resumed == fresh). Without either, ``same`` is "na".
--beside N: N other requests (short prompts, long replies) decode while each step after the first runs.
PASS: no request failed, every step after the first found cached tokens (unless --expect-cached 0), and with
--compare every reply equals the saved one. Standard library only; requests go to --url alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from longctx import OVERHEAD, PER_LINE, QUESTION, ask, document, emit, line_of  # noqa: E402

TOOL = "Tool result {k}: the next batch of station records follows.\n"
NOISE = "Count slowly from one to four hundred, one number a line, nothing else."


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--tokens", type=int, default=16000, help="about this many prompt tokens in the first step")
    p.add_argument("--grow", type=int, default=2000, help="about this many more in every later step")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--label", default="reuse")
    p.add_argument("--log", default="")
    p.add_argument("--save", default="")
    p.add_argument("--compare", default="")
    p.add_argument("--beside", type=int, default=0)
    p.add_argument("--expect-cached", type=int, default=1, help="0: the server keeps nothing (the fresh reference)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=900.0)
    a = p.parse_args()
    with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    lines = max(1, round((a.tokens - OVERHEAD) / PER_LINE))
    more = max(1, round(a.grow / PER_LINE))
    messages = [{"role": "user", "content": document(lines, a.seed) + QUESTION}]
    saved = json.load(open(a.compare)) if a.compare else None
    record, cached, firsts, same, failed = [], [], [], [], 0
    for k in range(1, a.steps + 1):
        noise: list = [None] * (a.beside if k > 1 else 0)

        def beside(i: int) -> None:
            noise[i] = ask(a.url, model, [{"role": "user", "content": f"({i}) " + NOISE}], 400, a.timeout)

        threads = [threading.Thread(target=beside, args=(i,)) for i in range(len(noise))]
        for t in threads:
            t.start()
        got = ask(a.url, model, messages, a.max_tokens, a.timeout)
        for t in threads:
            t.join()
        emit(line_of(f"{a.label}-s{k}", got), a.log)
        failed += sum(1 for g in noise if not g or "error" in g)
        if "error" in got:
            failed += 1
            break
        sha = hashlib.sha256(got["text"].encode()).hexdigest()[:16]
        record.append({"step": k, "prompt_tokens": got["prompt"], "completion_tokens": got["completion"],
                       "reply_sha": sha, "reply": got["text"]})
        cached.append(got["cached"] or 0)
        firsts.append(got["first"] - got["t0"])
        if saved is not None:
            ref = saved[k - 1] if k <= len(saved) else {}
            same.append(all(ref.get(key) == record[-1][key] for key in ("prompt_tokens", "completion_tokens", "reply")))
        messages += [{"role": "assistant", "content": got["text"]},
                     {"role": "user", "content": TOOL.format(k=k) + document(more, a.seed + 100 + k)
                      + "\n\nSay in two sentences what changed in this batch."}]
    if a.save:
        with open(a.save, "w") as fh:
            json.dump(record, fh, indent=1)
    reused = all(c > 0 for c in cached[1:]) if a.expect_cached else all(c == 0 for c in cached)
    ok = not failed and len(record) == a.steps and reused and all(same)
    emit(f"reuse_check label={a.label} steps={len(record)} beside={a.beside} "
         f"cached={','.join(str(c) for c in cached)} first_token_s={','.join(f'{t:.2f}' for t in firsts)} "
         f"same={'na' if saved is None else int(all(same) and len(same) == a.steps)} errors={failed} "
         f"result={'PASS' if ok else 'FAIL'}", a.log)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
