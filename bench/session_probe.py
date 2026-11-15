#!/usr/bin/env python3
"""Two agent sessions whose prompts do not fit the cache pool together (e.g. sessions of 235k and 101k tokens): their
steps alternate, so each step finds its own prefix evicted from the card by the other's step. With
TF_GLM_HOST_TIER_GB the prefix comes back from host memory; without it the step is read fresh.

    session_probe.py --url http://127.0.0.1:8000 --tokens 140000 --steps 4 --label L [--log F]
                     [--save FILE | --compare FILE] [--grow 2000] [--max-tokens 128]

Session A uses document seed 11, B seed 22, the same growth as reuse_check.py (the model's own replies, then a "tool
result"). One standard longctx line a step (label L-A1, L-B1, L-A2, ...: cached_tokens, first_token_s, prefill_tps
of the tokens read), then  session_probe label=L steps=.. cached=.. first_token_s=.. same=0|1|na result=PASS|FAIL
--save writes every step's prompt size and reply (run it on a load without the tier: every step read fresh, cached 0),
--compare checks them (resumed or restored == fresh). PASS: no error, and with --compare every reply the same and
every step after a session's first with cached tokens. Standard library only; requests go to --url alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from longctx import OVERHEAD, PER_LINE, QUESTION, ask, document, emit, line_of  # noqa: E402
from reuse_check import TOOL  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--tokens", type=int, default=140000)
    p.add_argument("--grow", type=int, default=2000)
    p.add_argument("--steps", type=int, default=4, help="steps a session")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--label", default="sessions")
    p.add_argument("--log", default="")
    p.add_argument("--save", default="")
    p.add_argument("--compare", default="")
    p.add_argument("--timeout", type=float, default=1800.0)
    a = p.parse_args()
    import urllib.request

    with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    lines = max(1, round((a.tokens - OVERHEAD) / PER_LINE))
    more = max(1, round(a.grow / PER_LINE))
    talks = {name: [{"role": "user", "content": document(lines, seed) + QUESTION}] for name, seed in (("A", 11), ("B", 22))}
    seeds = {"A": 11, "B": 22}
    saved = json.load(open(a.compare)) if a.compare else None
    record, cached, firsts, same, failed = [], [], [], [], 0
    for k in range(1, a.steps + 1):
        for name in ("A", "B"):
            got = ask(a.url, model, talks[name], a.max_tokens, a.timeout)
            emit(line_of(f"{a.label}-{name}{k}", got), a.log)
            if "error" in got:
                failed += 1
                break
            record.append({"step": f"{name}{k}", "prompt_tokens": got["prompt"], "completion_tokens": got["completion"],
                           "reply_sha": hashlib.sha256(got["text"].encode()).hexdigest()[:16], "reply": got["text"]})
            cached.append(got["cached"] or 0)
            firsts.append(got["first"] - got["t0"])
            if saved is not None:
                ref = saved[len(record) - 1] if len(record) <= len(saved) else {}
                same.append(all(ref.get(x) == record[-1][x] for x in ("prompt_tokens", "completion_tokens", "reply")))
            talks[name] += [{"role": "assistant", "content": got["text"]},
                            {"role": "user", "content": TOOL.format(k=k) + document(more, seeds[name] + 100 + k)
                             + "\n\nSay in two sentences what changed in this batch."}]
        if failed:
            break
    if a.save:
        with open(a.save, "w") as fh:
            json.dump(record, fh, indent=1)
    reused = all(c > 0 for c in cached[2:]) if saved is not None else True
    ok = not failed and len(record) == 2 * a.steps and reused and all(same)
    emit(f"session_probe label={a.label} steps={len(record)} cached={','.join(str(c) for c in cached)} "
         f"first_token_s={','.join(f'{t:.2f}' for t in firsts)} "
         f"same={'na' if saved is None else int(all(same) and len(same) == 2 * a.steps)} errors={failed} "
         f"result={'PASS' if ok else 'FAIL'}", a.log)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
