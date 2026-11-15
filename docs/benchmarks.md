# Benchmarks

All measured on one machine (8× RTX PRO 6000 Blackwell Server Edition, see the README) over
2026-11-13…15 UTC. Served build: the GLM-5.3 (`glm_moe_dsa`) TensorFold family with our NVFP4 / 8-rank /
tier / growth work, `--tp 8 --context 262144 --parallel 32 --mtp-drafts 2`, BF16 cache, greedy unless a
row says "sampled". Reasoning (thinking) is off in the raw speed probes and on in the conversation and
gr8bench runs, as noted. Figures labelled **K1 / D1 / W2 / M2** are named test loads; the first three of
these were the serving build's basis (K1 = the build that went live 2026-11-14).

Run-to-run spread below ~2 % is noise on this machine (confirmed by A/B loads that changed only NCCL
channel count or prefetch window and moved < 2 %). Worse results are kept here deliberately.

Methods, where they differ, are spelled out so a figure can be reproduced with the scripts in
[`bench/`](../bench/).

---

## 1. One request (single stream)

| Metric | Result | Load / date | Notes |
|---|---|---|---|
| Decode, short prompt, greedy | **157.4–158.1 tok/s** | K1/D1, 2026-11-14 | `speed_probe.py`, 384-token reply, thinking off, decode-only tok/s |
| Decode, short prompt, sampled | 153.1–153.5 tok/s | same | temperature 1.0, top-p 0.95, top-k 20 |
| First token, idle server | 0.067–0.068 s | same | first streamed piece |
| Round time | ≈ 15.0 ms (forward 13.0, drafts ~1.4, between ~0.6) | same | 2.42 tokens/round |
| Single-stream engine (one request at a time), 1/2/3 drafts | 143 / 170 / 174 tok/s | 2026-11-13 | a separate engine mode, not the serving config |

The serving build is **ahead of the public comparison for one request** (their 135–147, median 142;
see §6).

## 2. Several requests together

Two client methods, both reported:
- **`speedtest.py` / `wide_probe.py`** — 8 distinct prose/code questions, streamed, 1 round; total = all
  tokens over wall time from first send to last reply.
- **Like-for-like** — the public comparison's `bench_decode.py` method: N one-sentence prompts, not
  streamed, temperature 0, `ignore_eos`, 256-token replies, tokens over wall time with the prompt read
  inside. This is the figure to set beside the public 32-stream number.

| Streams | `speedtest` greedy | `wide_probe` greedy | like-for-like (256-tok) | Load |
|---|---|---|---|---|
| 2 | 209.6 | 209.3 | — | D1 |
| 4 | 334.4 | 332.8 | — | D1 |
| 8 | 496.8 | 507.6 | — | D1 |
| 16 | 674.2 | 709.3 | 753.5 | D1 |
| 32 | 968.7 | 1,011.5 | **1,039.8 / 1,051.0** | D1 |

- 32 together, 1,024-token replies: 1,019.5 (D1); steady part 1,013.9.
- 32 together, sampled (default sampling): 750.8 (D1).
- 32 together, no drafts: 766.9 (D1).
- **Soak**, 4 × (32 streams × 1,500 tokens): 924.5 / 914.4 / 924.9 / 936.8 (D1).
- The goal was ~1,000 tok/s at 32 together: **met** on the like-for-like method, the steady run and
  `wide_probe`; **not** on `speedtest` (969, 512-token replies) or the soak (912–937).

### Worse / superseded 32-stream figures (kept on purpose)

| Build | 32 together | Why worse |
|---|---|---|
| Served baseline before the wide reduce (`--parallel 32`) | 470–498 tok/s | decode windows >8 rows on the fp32 all-gather |
| W1 (first exact-path build, 8 NCCL channels) | 913.6 (speedtest), 994–997 (like-for-like) | before 16 channels |
| C16/W2 (16 channels) | 981 (speedtest), 1,037–1,052 (like-for-like) | the step to the served level |

**Exactness caveat on earlier 32-stream numbers:** before the exact row-blocks fix (W1, 2026-11-14), rounds
wider than 64 rows (23+ streams with drafts) were **not** bit-exact on these cards — a concurrent reply
could leave its solo reply at a near-tie token (measured: the first reply drifted at ~token 80 over
256 tokens). All 32-stream figures from before W1 are on that non-exact path. From W1 on, `together_check`
confirms 32 concurrent replies = the same 32 alone.

## 3. Long context (one request, decode after the prompt)

`longctx.py`, decode tok/s after the prompt is read, greedy.

| Context | K1 (served build) | D1 | W2 (before long-ctx graph capture) |
|---|---|---|---|
| 3,179 | **128.3** | 122.9 | 106.0 |
| 18,386 | **138.0** | 132.0 | 122.5 |
| 72,486 | **122.7** | 116.8 | 109.4 |
| 135,176 | **124.9** | 119.5 | 104.2 |

- Our own target was ≥ 150 tok/s on these lines; it was **not reached exactly**. The honest reading from the
  logs: ~120–138 is the ceiling for these prompts (2.1–2.3 tokens/round against 2.42 for a short prompt —
  the text drafts less well — plus forward 15.3–16.3 ms against 13.0 for the context). Agent-step replies in
  the same contexts (the 3rd step of a reuse check) already read 155–157 tok/s, because they draft better.
- Capturing long-context graphs at warm-up (D1 → K1) lifted the first-request lines to the replayed level.

## 4. Prefill (prompt read), alone

`longctx.py` first-token timing; tok/s = prompt tokens / time to first token.

| Context | tok/s | Load |
|---|---|---|
| 3,179 | 4,578 | D1 |
| 18,386 | 4,613–4,645 | D1 |
| 72,486 | 4,544 | D1 |
| 135,176 | 4,512 (first token 30.0 s) | D1 |

Prefill is **behind the public comparison** (their 6.3k–7.9k at 32k–262k). The limit here is the chain of
per-chunk reduces over PCIe host shared memory (≈ 4.4k tok/s floor with 8 NCCL channels, ≈ 5.1k with 16),
not the kernels; 6,000+ tok/s is not reachable exactly on this rank layout with this transport. See
[`design.md`](design.md).

## 5. Exactness and accuracy

| Check | Result |
|---|---|
| 32 concurrent replies vs the same 32 alone | **32/32 identical** (`together_check`), on every build from W1 on |
| Drafted reply vs serial (no-draft) reply | identical, token for token, every probe |
| Seeded reply repeats | yes; CUDA-IPC gather and NCCL give the same replies |
| CPU float32 reference ("golden") | argmax equal at **314 / 324** positions; the 10 differences are all near-ties (the reference leads by 0.03–0.89 logits) |
| GSM8K, full 1,319-question set, exact path | **97.19 %** (15 truncated), 2026-11-14 — the same figure the public comparison reports |

### The lossy FP4-experts experiment (parked)

A W4A4 kernel (routed experts on the card's block-scaled FP4 MMA) was built and measured against the exact
path on the same head:

- Prefill **+10.5 to +11.4 %**; decode within ±1 % at equal rows.
- But it **changes replies** (0 of 96 greedy replies identical; first difference median token 23.5), golden
  307/324, and **GSM8K 96.89 % vs the exact path's 97.19 %** (paired, McNemar p = 0.424 — not a significant
  accuracy loss, but the replies differ).
- Marked lossy, **off by default**; it never shows as a "best" result. The served path is exact.

## 6. Comparison with a public vLLM-based setup

Against **`kacper-daftcode/GLM5.3-8xPRO6K`** (Apache-2.0; patched vLLM nightly, 4 tensor-parallel × 2
pipeline stages, 3 MTP drafts, `incoai/GLM-5.3-NVFP4` with BF16 layers turned to FP8 at load and an NVFP4
cache — **not exact**; results dated 2026-09-17…22, same class of machine: 8× RTX PRO 6000 Blackwell,
PCIe-only, 2 NUMA nodes). Their raw logs were read, **nothing of theirs was run.**

| Metric | Ours | Theirs | Comparable? |
|---|---|---|---|
| One request, greedy | 157.6 (≈ 154 counted their way) | 135–147, median 142 | Partly — ours decode-only/thinking-off, theirs whole-request/thinking-on, lossy weights; **ours faster (~+8–11 %)** |
| Round time at 3 drafts | 16.5 ms | 21.1–21.3 ms | Yes for the engine step (with the quantization caveat) |
| 4 together | 245 | 300–311 | Mostly — **theirs faster (+23–27 %)** |
| 16 together | 710–753 | 654–666 | Roughly — close/ahead |
| 32 together | 1,040–1,051 (like-for-like) | 1,041–1,056 | **Level** |
| Prefill, 32k/131k/262k | not at this scale (~4.6k at ≤128k) | 7.9k / 7.2k / 6.3k | **Theirs faster** |
| Context per request / pool | 262,144 shared | 750,000 / 1.81M pool | Different designs; **theirs larger** |
| Start to ready | ~350 s (O_DIRECT) | ~140 s warm page cache (105–400 s range) | **Theirs faster** (warm cache vs our deliberate O_DIRECT) |
| GSM8K (1,319) | 97.19 % (exact) | 97.19 % (97.50 % without their extra quant) | Level |
| Reproducible replies | yes (exact) | no (same prompt gives other greedy text cold/cached/reloaded) | **A real difference in our favour** |
| Cache bytes/token (8 cards) | ~122 KiB (BF16) | ~120 KiB (4-bit cache, ~8× smaller per layer) | theirs more compact |

**Fair caveats:** different weights of the same family; they run FP8 dense + 4-bit cache (lossy), we run the
checkpoint's exact bits; they include time-to-first-token in the decode figure and we do not; their image
cannot be rebuilt from public sources (pinned base). Full table and per-figure sourcing were produced from
their public `results/` logs.

**Where each wins:** we are faster and exact for one request and level at 32 together; they win on prefill,
larger context, faster warm start, and raw concurrency ramp (their TP4×PP2 pipeline keeps each exchange
among 4 cards of one NUMA node). A paper estimate of adopting their TP4×PP2 layout on our stack came out
*worse* for ≤32 streams on this hardware (experts' bytes don't shrink with the split), so it was not built.

## 7. gr8bench-style run (serving instance, 2026-11-14 10:55–11:10 UTC)

Our own code, method modelled on **gr8bench** (`kindlingai/gr8`, MIT); 32 lanes, the served build. 0 of
1,848 health samples showed another user — every timed figure is undisturbed.

- **One stream by text kind** (512-token replies, effort low, gr8's prompts — these draft well):
  code 184.5 / prose 145.5 / structured 194.1 / counting 197.8 tok/s.
- **Together 1/2/4/8/16/32:** 172.9 / 216.7 / 341.2 / 490.0 / 663.4 / **961.0** tok/s.
- **Prefill** 32K / 64K / 128K: 4,623 / 4,601 / 4,421 tok/s; the repeated prompt reuses whole 8,192-token
  blocks (cached first-token 0.145 / 0.157 / 0.238 s).
- **Needle** at 32K / 128K / 250K (256,350 tokens, first token 61.5 s): PASS. (256K+ don't fit our 262,144
  context.)
- **Agent tool tasks** 10/10 PASS; count-to-200 5/5; **GSM8K first 250: 96.8 %**; 75-item quality check
  74/75 each round (the one miss, `json-extract`, is a correct answer the model left inside its reasoning).
- The gr8 smoke stage "fails" are API-convention differences (forced tool calls may still think first;
  logprobs/refusal wording), not engine faults — listed in the run report.

> The ledger's "one request 197.8" best comes from these easy-to-draft prompts and is **not** comparable
> with the standard probe (158).

## 8. Conversation speed (4-turn coding chat, serving model)

`conv_speed.py`, client defaults, streamed, max_tokens 32,000, no other users.

| Effort | Reply tokens, turns 1–4 | Reasoning chars 1–4 | Answers | Overall tok/s 1–4 | Wall (4 turns) |
|---|---|---|---|---|---|
| **max** (template default) | 14,620 / 32,000 / 32,000 / 32,000 | 47,105 / 112,132 / 112,246 / 132,910 | 1 whole, 2 cut, 1 none | 149 / 137 / 138 / 126 | ~21 min |
| **high** | 759 / 5,248 / 3,548 / 12,170 | 392 / 130 / 1,425 / 514 | all 4 whole | 163 / 144 / 150 / 141 | ~2.5 min |
| **low** | 688 / 2,769 / 2,798 / 9,675 | 0 / 30 / 212 / 89 | all 4 whole | 179 / 152 / 145 / 138 | ~2 min |

No large speed drop as context grows: the forward per round is flat (14.69 → 14.91 ms from turn 1 to 4, for
~40k tokens of added context). The spread of tok/s (118–183) follows draft acceptance (code drafts well,
long prose reasoning less). **The real finding is about effort, not speed:** at `max` the model reasons
100–1,000× longer than at `high` and runs into the token cap — this is why the recipe defaults to `high`.

## 9. Host-RAM prompt tier (repeated long contexts)

Re-check window 2026-11-15 05:07–05:21 UTC (`session_probe.py`, two ~140k-token agent sessions alternating,
8 steps):

| | Without the tier | With the tier (40 GiB/rank) |
|---|---|---|
| First token per step | 39.6 s every step | 39.6, 39.6, 2.3, 2.2, 2.8, 2.8, 1.4, 1.4 s |
| Replies | — | identical (`same=1`), cached 172k–180k tokens |

Fairness alongside (`fair_probe.py`, 4 decoders beside a 164k read): chunks/s inside the window 61.4 → 80.6,
largest decoder gap 1.88 → 0.24 s; the long read's own first token 61.4 → 62.3 s (unchanged). A short request
beside the read still gets its first token in 0.23 s. Exactness held: `together_check` 32/32, golden 314/324.

NVMe as a third tier was considered and declined for this disk (a 140k session ≈ 109 GB ≈ 16 s to restore,
and the write wear is high under heavy use); watch the RAM tier's hit rate first.

## 10. Growing reply room + streamed tool calls

Re-check window 2026-11-15 09:19–09:37 UTC (`grow_probe.py`):

- A **250,724-token** prompt whose `max_tokens` fills the context runs while **6 short requests** start in
  **0.07–0.09 s** (one 6.4 s, hit while the long prompt was still being read); the long reply equals its
  solo reply; 0 growth waits, 0 streams gave way. Before this, a short request behind such a prompt could
  wait ~145 s (the whole long request).
- A forced **tool call** streams as **809 argument fragments over 12.4 s**, byte-equal to the whole call,
  on both `/v1/chat/completions` and `/v1/responses`.

---

### Provenance

Every number above is from this machine's test logs on the dates given. The named loads (K1, D1, W2, M2,
W1, C16) are development test loads; the serving build went live 2026-11-14 on the K1 head and gained the
host-RAM tier (2026-11-15 05:29) and growing reply room + streamed tool calls (2026-11-15 09:37) in later
windows, each after its checks passed. The public-comparison figures are from that project's committed
`results/` logs, read read-only.
