# Design notes: what we changed on top of the GLM fork, and why

Prose only — **no engine source is in this repository.** This explains the changes that produced the
numbers in [`benchmarks.md`](benchmarks.md), on top of the upstream GLM-5.3 (`glm_moe_dsa`) CUDA family in
the Python TensorFold fork `bertholomus/TensorFold@glm-dsa-tp4` (Apache-2.0). The engine changes themselves
are not public yet.

The starting point: the upstream fork serves the full GLM-5.3 on CUDA, but was validated on 4× DGX Spark
(GB10) with EXL3 weights (one stream ~33–43 tok/s, prefill ~400 tok/s), and only for EXL3. Our checkpoint is
**ModelOpt NVFP4**, and our machine is **8 local RTX PRO 6000 on PCIe, no NVLink**. So the work was: read
this checkpoint exactly, run it on 8 local ranks, and make it fast on this interconnect — without ever
changing a served reply's bits.

## Loading the NVFP4 checkpoint at tensor-parallel 8

- A loader for the checkpoint's exact ModelOpt NVFP4 layout (routed experts: `weight` as packed e2m1 codes,
  `weight_scale` per 16 inputs in F8_E4M3, a per-matrix F32 `weight_scale_2`; `input_scale` is exactly 1.0
  everywhere). The gate refuses anything outside that layout rather than guessing.
- Every rank holds **all 256 routed experts of a layer at its 1/8 slice of the width** — gate/up output rows
  for the rank only, `down` cut by input columns, each slice a whole 32-column block and a whole 16-input
  scale group. ≈ 48 GiB of experts + ≈ 7 GiB of BF16 (attention, shared experts, indexer, head) per card.
- The MTP (draft) layer's experts are BF16 on disk; they are quantized to NVFP4 at load (≈ 9.5 % relative
  weight error) — this affects only how many drafts are accepted, never a verified reply.
- **A latent-attention correctness bug in the fork, found and fixed:** with plain-BF16 attention (our
  checkpoint, vs the fork's trellis-group EXL3 path), the main and side CUDA streams shared one split-K
  scratch buffer in decode windows, so concurrent latent projections came back with wrong bits. Separate
  buffers fixed it (verified: 300/300 concurrent pairs correct).
- **Two RMS-norm epsilons were settled** against a CPU reference and against what vLLM/SGLang/exllamav3 do:
  the `q_a`/`kv_a` latent norms use the config's `rms_norm_eps` (what the checkpoint's card targets); the
  indexer's `k_norm` uses 1e-6 (which matters only past 2,048 tokens). The quality cost of either choice is
  below the BF16 rounding noise.

A CPU float32 "golden" forward of the real checkpoint (one layer in memory at a time, HF's own `GlmMoeDsa`
code as the source of truth) is the accuracy reference the GPU engine is checked against end to end and
layer by layer.

## Faster decode kernels

Profiling one decode round (3,211 kernel launches/forward, ~0.26 ms/layer idle) showed the routed experts'
gate/up + SwiGLU was ~half the kernel time, dominated by one warp per (expert, 32 columns) waiting on its
own loads (~158 µs/launch). A new decode kernel that stages weights and inputs through shared memory brought
that to ~24 µs/launch at 1 row — **bit-identical** to the old kernel (prompt chunks keep the old path). With
a couple of smaller changes (skipping group sums a BF16 `o_proj` never reads), the forward dropped from
~20.8 ms to ~10.5 ms per layer-pass, roughly doubling one-request throughput before any interconnect work.

## The interconnect: making 8 PCIe cards exchange cheaply

A decode round does ~157 small all-gathers of fp32 partials between the 8 ranks. On this machine that is the
cost that matters, and the lessons were counter-intuitive:

- **Host shared memory beats GPU-to-GPU for tiny payloads.** Grouped NCCL send/recv through host shared
  memory with **8–16 channels** is the cheapest small-gather (16.6 µs at 1 row); the GPU-to-GPU transport
  (`NCCL_P2P_LEVEL=SYS`) is 4–10× slower for these sizes and *stalls* concurrent requests, because the fixed
  per-exchange software cost dominates, not PCIe bandwidth. 16 NCCL channels (over 8) gave a clear win at
  width (32 together 914 → 993 tok/s on one probe).
- **A CUDA-IPC peer exchange for the small decode windows.** For windows up to 8 rows, the ranks exchange
  through CUDA-IPC-mapped peer memory instead of NCCL. The key was a store pattern where **each GPU writes
  into one peer at a time**: naive "every rank stores to all 7 peers at once" collapsed to ~1–3 GB/s on this
  hardware (the same collapse NCCL's GPU-to-GPU send/recv shows), while one-peer-at-a-time stores run at
  ~38 GB/s. Done this way the IPC gather beats NCCL for a lone request (+12–13 %) and needs a rank to see
  only its own GPU. NCCL still carries everything wider. *(This peer-exchange scheme follows a design shared
  privately with the project; its implementation is not published here.)*
- **L2 weight prefetch during exchange waits.** While a rank waits in an exchange, the weights the next
  kernels will read are pulled into the card's L2 (it writes nothing, so every reply keeps its bits). The
  waits are ~6 ms/forward with the card's memory otherwise idle; prefetching into the ~100 MB of usable L2
  added +14–17 % on the single-stream engine on top of the IPC gather. (Ported from gr8's `l2pf` kernel,
  MIT, notice kept in the engine.)

## Exactness at width — two fixes that cost nothing

- **Exact wide reduce.** Decode windows of 9+ rows (several requests' rounds) originally took the fp32
  all-gather (172 kB/row to a rank). They now take the same exact rows-reduce the prompt chunks use (32 kB/row,
  identical bits). This is the single biggest concurrency win: 16 together 387 → 610–657, 32 together
  470 → 800–887 tok/s, with the lone request unchanged.
- **Exact 64-row blocks.** On these cards the latent absorb/expand row-block kernels differed from the
  single-row kernel by one BF16 ulp on a few elements in a million once a forward exceeded 64 rows — enough
  that, at 23+ concurrent streams with drafts, a reply drifted from its solo run over a couple hundred
  tokens. The fix runs those kernels 64 rows at a time for wide forwards; 0 elements differ at 65…1,000 rows,
  at the cost of one extra launch per kernel per layer for 65–128 rows. A new `together_check` runs the 32
  served prompts alone and together on every build and requires the replies to match. (This is why some
  pre-fix 32-stream numbers in the benchmarks are flagged as non-exact.)

## Drafts

The model's own MTP head drafts. Measured: at the serving configuration **2 drafts/round is best** (3+ is
slower at width). External drafters (DSpark, DFlash2) were measured offline on this checkpoint and **do not
beat the MTP head** at the depths that pay (2–3), and lose badly on non-English text; neither was built. A
stronger drafter would not reach the 150-tok/s one-request target anyway — at ~3.3 tokens/round that needs a
~22 ms round, and kernel + exchange time is the real floor.

## Batched short-prompt fill

Short fresh prompts (≤ 63 tokens) now fill together — up to 432 rows per forward, bounded by the memory plan
beside the 262,144-token pool and 16 NCCL channels — so 32 short prompts arriving at once are ~3 forwards
instead of 16–32. This is what let `speedtest`-style concurrency figures rise without touching the
like-for-like number (short prompts were only slow because of the old 64-row limit).

## Keeping prompts: reuse, host tier, growth

- **Prompt reuse.** A finished request's prompt stays in the shared pool; a later request that starts with it
  reads only the new tokens. With the extents layout a kept prompt is cut to its own rows, grows in place or
  moves inside the pool as its next step lengthens, and gives way oldest-first under pressure. Reuse is by
  whole 8,192-token chunks. A growing conversation's later steps answer in 1–2 s instead of 5–23 s.
- **Host-RAM tier.** With two large agent sessions alternating, the 262k pool holds only one at a time, so
  each step evicted the other and re-read ~140k tokens (~40 s). The tier writes kept-prompt chunks back to
  **page-locked host RAM** (40 GiB/rank = 320 GiB of the 751 GiB box, ~440k tokens) on a side stream and
  restores them by H2D instead of re-reading — an evicted prompt returns in 1.4–2.8 s. Every write-back that
  a later read depends on is guarded so rows are never overwritten before they are saved; every rank makes
  the same tier decisions in the same order, so the ranks stay in agreement and replies are unchanged.
- **Growing reply room.** A stream is admitted with its prompt + 4,096 reply rows (not its whole max_tokens),
  and grows before each round — in place, or by moving the cheaper neighbour, or by evicting kept extents,
  oldest-first. So one 250k-token request no longer seizes the whole pool; short requests beside it still
  start in ~0.07–0.09 s. A stream that cannot grow waits with its state untouched (its reply stays bit-for-bit
  what it would be alone); if the pool is truly full, the newest stream gives way, returns to the queue and
  replays from its prompt (already-sent tokens are checked, not re-sent — the client sees a pause, never a
  changed or cut reply). `max_tokens` stays the hard cap.

## Fairness, long context, and prefill

- **Fair prefill.** A long prompt is read 8 layers at a time between decode rounds (fewest rows first), so
  decoders beside a 164k read keep 80.6 chunks/s instead of 61.4 and their largest stall drops from 1.88 s to
  0.24 s — the long read's own first token barely moves. At a stop boundary each micro-batch's pending work is
  folded into the next layer's norm (the same arithmetic), so nothing is left in flight on the comm stream;
  then the same kernels run on the same rows for the same bits.
- **Selection share.** A long prompt's sparse-token selection is computed once per rank-share (from 16,384
  tokens on, where it starts to weigh) instead of redundantly on every rank — exact, and worth +~14 % prefill
  at 72k.
- **Sparse-attention staging.** The long-context decode path loads 5 key tiles ahead instead of 3 (same
  tiles, same fold order, same bits) — +4–5 % long-context decode.
- **Prefill is transport-bound, not kernel-bound.** The chain of ~312 per-chunk reduces over PCIe host shared
  memory is a floor of its own (~4.4k tok/s at 8 NCCL channels, ~5.1k at 16), however fast the kernels get.
  Reaching 6,000+ tok/s would need fewer bytes on the wire (BF16 partials — lossy) or a different rank layout
  (a two-stage pipeline per NUMA node), neither of which was taken: the first changes bits, the second was
  estimated *worse* for ≤32 streams on this hardware (experts' bytes don't shrink with the split).

## Operational pieces

- **Graph management:** round and draft-step graphs are kept by last use with raised caps (so a stream with a
  long context always gets a graph — an earlier defect capped it and dropped such streams to ~40 tok/s);
  long-context round graphs are captured at warm-up so the first long request doesn't pay a capture cost
  mid-decode.
- **Lost-rank watchdog:** if a rank dies, the others end the request within ~1–3 s with a clear message and
  free the cards — no host power-cycle needed.
- **Parked, lossy, off by default:** FP4 experts (W4A4) — faster prefill but changes replies and GSM8K
  97.19 % → 96.89 %; FP8 dense projections — ~+6 % but changes ~1 token in 40. The served path keeps the
  checkpoint's exact BF16-dense, BF16-cache bits.

## What is still open

- Long-context decode is ~120–138 tok/s, under our 150 target (the ceiling is the text's lower draft
  acceptance plus ~2.5–3 ms more forward, not a bug).
- Prefill is ~4.6k tok/s, below comparable vLLM setups, bounded by the transport as above.
- Not measured here: compact cache formats and 1M-token context (decode-context-parallel) on the real model;
  a multi-hour mixed soak; power-limit effects on these Server-Edition cards.
