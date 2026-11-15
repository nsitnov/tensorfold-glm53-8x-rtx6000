# Recipe: GLM-5.3 NVFP4 on 8× RTX PRO 6000 Blackwell with TensorFold

Step by step. This serves the `dealignai/GLM-5.3-UNCENSORED-NVFP4` checkpoint as 8 local tensor-parallel
ranks (one GPU each) behind the engine's own OpenAI-compatible HTTP server.

> The engine that implements the NVFP4 / 8-rank / tier / growth work is **not published here.** Build it
> from the upstream GLM fork (see the README credits) plus the changes described in
> [`design.md`](design.md). Everything else below — platform setup, model, launch flags, serving settings
> and their measured effect, checks — is complete.

Paths in this document are neutral placeholders (`/models/...`, `/opt/tensorfold`). Use your own.

---

## 1. Platform (BIOS / OS / kernel)

The hardware this was measured on: 8× RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W, sm_120, PCIe
Gen5 x16, **no NVLink**), 2× AMD EPYC 9555 (2 NUMA nodes; GPUs 0–3 on node 0, GPUs 4–7 on node 1), 751 GiB
RAM, the checkpoint on a PCIe Gen4 NVMe.

**The one setting you must not skip: `iommu=pt`.** On this AMD platform with the IOMMU in its default
(DMA-FQ / translated) mode, **peer-to-peer copies between GPUs are silently corrupted** — a bf16 matmul
done cross-card comes back wrong with no error. Pass-through mode fixes it:

```
# kernel command line
iommu=pt
```

Add it (e.g. a GRUB drop-in that appends `iommu=pt` to `GRUB_CMDLINE_LINUX_DEFAULT`), regenerate the
bootloader config, reboot, and confirm:

```bash
grep -o 'iommu=pt' /proc/cmdline            # expect: iommu=pt
```

A quick cross-card sanity check after boot: a bf16 matmul on each card and a device-to-device copy between
every pair must match a CPU reference. Until `iommu=pt` is in effect, multi-GPU work is unsafe; a single
card is fine, and NCCL can be forced through host memory with `NCCL_P2P_DISABLE=1` as a stopgap.

Other platform notes:
- Confirm the PCIe/NUMA layout with `nvidia-smi topo -m` (expect `NODE` within a socket, `SYS` across).
- No container, no root, no NVLink are required. The engine runs as an ordinary user.

## 2. Driver, CUDA, PyTorch

- Driver **595.91.07** (reports CUDA 13.2 to the runtime).
- CUDA **13.1** toolkit — `nvcc` V13.1.115 — used to compile the engine's CUDA/Triton kernels for sm_120.
- **PyTorch 2.15.0+cu132** (the cu132 wheel ships its own CUDA runtime, cuDNN 9.26, NCCL 2.30.7),
  Triton 3.9.0, Python 3.12, in a dedicated virtualenv.

```bash
python3.12 -m venv /opt/tensorfold/venv
/opt/tensorfold/venv/bin/pip install torch==2.15.0+cu132 \
    --index-url https://download.pytorch.org/whl/cu132
```

Pin Triton's fp-fusion to the 3.9 default explicitly (`TRITON_DEFAULT_FP_FUSION=1`) — Triton 3.10 changes
it, which would change kernel bits.

## 3. Model

Download `dealignai/GLM-5.3-UNCENSORED-NVFP4` (≈ 433 GB, 282 safetensors shards) to fast local storage,
e.g. `/models/GLM-5.3-UNCENSORED-NVFP4`. Verify it:

- Check every shard against `model.safetensors.index.json` (all tensor names present, shapes and dtypes as
  the index states) and against the publisher's manifest/hashes if you have them.
- The checkpoint is **ModelOpt NVFP4** for the routed experts (`weight` as U8 holding two e2m1 codes per
  byte, `weight_scale` F8_E4M3 per 16 inputs, `weight_scale_2` F32, `input_scale` F32 = 1.0 everywhere);
  everything else (attention, shared experts, indexer, head, MTP layer) is BF16. A loader that cannot read
  exactly this layout should refuse rather than guess.
- It is **text only** — there are no vision tensors; do not expect image input.

### Optional: pre-split per rank

A cold start reads every `down_proj` whole on each of the 8 ranks (≈ 191 GiB/rank, ≈ 1.5 TiB total) because
each rank keeps all 256 experts at its slice of the width. Writing per-rank folders once (each rank's own
NVFP4 bytes, ≈ 480 GB on disk in total) lets each rank read only its share and cuts the start from ~6 min to
an estimated ~1.5–3 min. It costs the extra disk and was **not used** in the measured runs here (the start
figures below are the full-checkpoint path, read with O_DIRECT so it does not evict other page-cache users).

## 4. Launch

8 ranks, rank *R* on the *R*-th GPU in PCI-bus order, each pinned to the CPUs of its GPU's NUMA node, each
seeing **only its own GPU** (`CUDA_VISIBLE_DEVICES`-style isolation). The rendezvous and NCCL bootstrap stay
on loopback (`127.0.0.1`). The served configuration:

```
--tp 8 --context 262144 --parallel 32 --mtp-drafts 2
```

- **`--tp 8`** — tensor parallel across the 8 cards (heads 64/8, expert width 2048/8, vocab 154,880/8).
- **`--context 262144`** — one 262,144-token pool shared by all streams; fits with the BF16 cache at
  ≈ 88–92 GiB/card.
- **`--parallel 32`** — up to 32 requests decoded together.
- **`--mtp-drafts 2`** — 2 MTP draft tokens per round (measured best for the serving configuration; 3+ is
  slower here, see the benchmarks).
- Memory reserve: keep ~6 GiB/card free (`reserve 6 GiB`); the default (a tenth, ~9.5 GiB) would refuse the
  262,144 window with the BF16 cache.
- Serve on a **local port** — the engine's own OpenAI-compatible server, e.g. `127.0.0.1:8000`. There is no
  gateway or key layer in this recipe; put one in front yourself if you need auth.

### Reasoning effort = high

Default the server to `reasoning_effort high`. This checkpoint only honors `low` and `high` (everything else
becomes `max`), and at `max` the model reasons enormously: a 4-turn coding conversation produced 47k–133k
**characters** of reasoning per turn and hit a 32,000-token cap in 3 of 4 turns (one with no answer at all).
At `high` all four turns answered, with 130–1,425 characters of reasoning. For minimum latency, pass `low`.

## 5. Serving settings that matter (measured effect)

These are set in the environment before launch; each keeps the checkpoint's exact bits unless marked lossy.
Numbers are from this machine (see [`benchmarks.md`](benchmarks.md) for the full runs).

| Setting | Value used | What it does | Measured effect |
|---|---|---|---|
| CUDA-IPC decode gather | on, ≤ 8 rows | Peer exchange for small decode windows through CUDA-IPC-mapped memory instead of NCCL; NCCL for everything wider | One request 142.8 → **157.6** tok/s (serving, with L2PF); exact, replies unchanged |
| L2 weight prefetch | on | While a rank waits in an exchange, the weights the next step reads are pulled into L2 | Single-stream +14–17 % (e.g. 128→143 / 151→170 / 158→174 at 1/2/3 drafts); writes nothing |
| Exact wide reduce | on, ≥ 9 rows | Decode windows of 9+ rows exchange via the prompt-chunk rows reduce (32 kB/row to a rank) instead of the fp32 all-gather (172 kB/row), same bits | 16 together 387→**610–657**, 32 together 470→**800–887** tok/s |
| 16 NCCL channels | 16 | Grouped send/recv through host shared memory with 16 channels | 32 together 966→**981** (speedtest), 914→**993** (wide_probe); 18k prompt prefill 4,092→**4,629** |
| Exact row-blocks | on | Forwards wider than 64 rows run the latent absorb/expand 64 rows at a time, so a concurrent reply is bit-identical to its solo run (without this, 32-stream replies drifted) | 32 concurrent **= solo 32/32**; no measurable cost |
| Batched short-prompt fill | 432 rows | Short fresh prompts (≤ 63 tokens) fill together, up to 432 rows/forward, instead of one at a time | 32 arriving at once: 3 forwards not 16–32; first piece at 32 together 1.64 → 0.45 s |
| Prompt reuse | on | A finished prompt stays in the pool; a later request starting with it reads only what is new | A growing conversation's later steps: first token 5–7 s → **1.4–1.9 s** (18k) and 20–23 s → **1.1–2.5 s** (72k), same replies |
| Host-RAM prompt tier | 40 GiB/rank | Evicted kept-prompt chunks are written to page-locked host RAM and restored by H2D instead of re-read (320 GiB pinned = ~440k tokens beside the cards' 262k) | Two ~140k sessions alternating: first token 39.6 s → **1.4–2.8 s**, replies identical |
| Fair prefill (fill in slices) | 8 layers, fewest-rows-first | A long prompt is read 8 layers at a time between decode rounds | Decoders beside a 164k read: 61.4 → **80.6** chunks/s, largest gap 1.88 → **0.24 s**; the long read's first token 61.4 → 62.3 s |
| Selection share | on, from 16,384 | A long prompt's token selection is computed once per rank-share instead of on every rank, exact | 72k prompt prefill 4,000 → **4,546** tok/s |
| Growing reply room | on, +4,096 rows | A stream is admitted with prompt + 4,096 reply rows and grows as it writes (max_tokens stays the cap), so one long request no longer takes the whole pool | A 250k-token request runs while 6 short requests start in **0.07–0.09 s**; replies equal |
| Streamed tool calls | on | Tool-call arguments stream as OpenAI deltas instead of one chunk at the end | 800+ fragments over ~12 s, byte-equal to the whole call |
| **FP4 experts (W4A4)** | **off (lossy)** | Routed experts on the card's block-scaled FP4 MMA; speeds prefill | +10–11 % prefill but **changes replies**, GSM8K 97.19 % → 96.89 %; **parked** |
| **FP8 dense projections** | **off (lossy)** | BF16 projections as FP8 | +~6 % for one request but changes ~1 token in 40; **off** |

The cache format is BF16 (the reference bits). Compact caches (fp8, q8…q4) and 1M context via
decode-context-parallelism exist in the engine but were not measured here and are not exact against BF16.

## 6. Warm-up, start time, health checks

- **Start to ready:** ≈ 350–377 s. Weights read in ≈ 328–330 s per rank with O_DIRECT (so the read does not
  evict other page-cache users); engine + graph capture add a few seconds. Long-context round graphs are
  captured at warm-up so the first long request does not pay a capture cost mid-decode.
- **Health:** the server exposes `/v1/models` (lists the served model id), `/v1/chat/completions`,
  `/v1/completions`, `/v1/responses`, and `/health` (running/decoding counts, prompt/cached token totals).
- **Checks to run after a start** (all in [`bench/`](../bench/)):
  - `together_check.py` — 32 prompts one at a time, then all 32 at once; every reply together must equal
    the reply alone (expect `same=32/32`). This is the exactness gate.
  - `speed_probe.py` — one-request decode speed and the per-round time breakdown.
  - `speedtest.py` / `wide_probe.py` — N requests together.
  - `longctx.py`, `reuse_check.py`, `session_probe.py` — long context, prompt reuse, host-tier restore.
  - `fair_probe.py`, `grow_probe.py` — fairness under a long prompt, and growing reply room + streamed
    tool calls.
  - A CPU float32 reference ("golden") of the same checkpoint is the accuracy gate used during development
    (argmax equal at 314/324 positions, the 10 differences all near-ties); building that reference is part
    of the engine work and is not included here.

See [`bench/README.md`](../bench/README.md) for how to run each against any OpenAI-compatible endpoint.
