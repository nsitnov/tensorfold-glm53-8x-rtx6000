# tensorfold-glm53-8x-rtx6000

Serving **GLM-5.3** (the `dealignai/GLM-5.3-UNCENSORED-NVFP4` checkpoint) with a **TensorFold**-based
engine on **8× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)**, PCIe-only, no NVLink — a recipe
and a set of benchmarks from one machine.

This repository is **a recipe and measurements, not an engine release.** The CUDA/Python engine work that
produced these numbers — native ModelOpt-NVFP4 expert loading at tensor-parallel 8, a CUDA-IPC peer
exchange over PCIe, L2 weight prefetch, an exact wide-decode reduce, a host-RAM prompt tier, growing reply
room — **is not published here yet.** The docs describe *what* was changed and *why*, and link the upstream
projects it builds on; they contain **no engine source.** If you want to reproduce the serving side today,
start from the upstream fork linked in [Credits](#credits) and the design notes in
[`docs/design.md`](docs/design.md).

What you can run from this repo is the **client-side benchmark and test harness** in [`bench/`](bench/):
standard-library Python that talks HTTP to any OpenAI-compatible endpoint.

---

## Headline numbers

One machine, the served build (`--tp 8 --context 262144 --parallel 32 --mtp-drafts 2`), greedy unless
noted. Every figure is sourced in [`docs/benchmarks.md`](docs/benchmarks.md); the worse results and the
caveats are there too.

| What | Result |
|---|---|
| One request, short prompt, decode | **≈ 158 tok/s** (157.4–158.1), first token ≈ 0.068 s |
| 16 requests together | **≈ 710–753 tok/s** |
| 32 requests together | **≈ 1,040–1,050 tok/s** (256-token replies, like-for-like method); ≈ 1,010 on a longer-reply probe |
| Long context, one request, decode | **128 / 138 / 123 / 125 tok/s** at 3k / 18k / 72k / 135k tokens |
| Prefill (prompt read), alone | **≈ 4,600 tok/s** at 18k–128k |
| GSM8K, full 1,319-question set | **97.19 %** (exact, non-lossy path) |
| Exactness | 32 concurrent replies **= the same 32 replies run alone** (bit-for-bit); drafted = serial; matches a CPU float32 reference at 314/324 near-tie-free positions |
| Repeated long context (host-RAM tier) | an evicted ~140k prompt returns in **1.4–2.8 s** instead of ~40 s |
| Growing reply room | a 250k-token request can run while short requests still start in **0.07–0.09 s** |
| Context | one **262,144-token** pool shared by up to 32 streams |
| Start to ready | ≈ 350–377 s (weights read with O_DIRECT) |

Reasoning (thinking) is on by default at reasoning effort **high**; see the recipe for why.

---

## Hardware

- **GPUs:** 8× NVIDIA RTX PRO 6000 Blackwell **Server Edition**, 96 GB each (97,887 MiB), 600 W, sm_120,
  one tensor-parallel rank per GPU.
- **Interconnect:** PCIe Gen5 x16 per GPU, **no NVLink.** GPUs 0–3 on NUMA node 0, GPUs 4–7 on node 1;
  cross-node topology is `SYS` (`nvidia-smi topo -m`), so inter-GPU traffic crosses the host bridges.
- **CPU:** 2× AMD EPYC 9555 (128 cores / 256 threads total, 512 MiB L3, 2 NUMA nodes).
- **RAM:** 751 GiB.
- **Storage:** the 433 GB checkpoint on a 1.92 TB PCIe Gen4 NVMe SSD; OS on SATA SSDs.

See [`docs/recipe.md`](docs/recipe.md) for the BIOS/kernel settings that turned out to matter (notably
`iommu=pt` — without it, inter-GPU copies on this platform are silently wrong).

## Software

- Driver **595.91.07** (the driver reports CUDA 13.2), CUDA **13.1** toolkit (`nvcc` V13.1.115).
- **PyTorch 2.15.0+cu132** (ships its own CUDA runtime, cuDNN 9.26, NCCL 2.30.7), Triton 3.9.0,
  Python 3.12. Linux 7.0.0.
- Engine: the GLM-5.3 (`glm_moe_dsa`) family of **TensorFold** (Python fork
  `bertholomus/TensorFold@glm-dsa-tp4`, Apache-2.0) with our own NVFP4 / 8-rank / tier / growth work on top.
  **That engine work is not in this repository** (see the note at the top).

## The model

`dealignai/GLM-5.3-UNCENSORED-NVFP4` — a **weight-level uncensored (abliterated) build of
[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)** in NVFP4, released by dealignai under the
**MIT license** (see the model card). Key facts used here, read from the checkpoint and its card:

- Architecture `glm_moe_dsa` — MoE + MLA + DeepSeek sparse attention; **753B total, ~18B active** per
  token; 78 layers + 1 MTP (multi-token-prediction) draft layer.
- Quantization: routed experts in **NVFP4** (ModelOpt layout); attention, shared experts, indexer, head and
  the MTP layer in BF16. On disk ≈ 433 GB, ≈ 55.3 GiB of weights per card at tensor-parallel 8.
- 1M position embeddings; `reasoning_effort` honors only `low` and `high` (everything else falls through to
  `max`); the MTP draft head is also uncensored (card: ~87 % draft acceptance).
- **Text only** — the checkpoint has no vision tensors; image input is not possible with this model.

The checkpoint is a red-team / offensive-security model (its card states this plainly). It is served here for
security research; this repository is only the serving recipe and the measurements.

---

## Recipe overview

Full step-by-step in [`docs/recipe.md`](docs/recipe.md). In short:

1. **Platform:** install the driver/CUDA/PyTorch stack above; set `iommu=pt` on the kernel command line
   (mandatory on this AMD platform for correct peer copies); confirm PCIe topology.
2. **Model:** download `dealignai/GLM-5.3-UNCENSORED-NVFP4` to fast local storage; verify the shards.
3. **Engine:** build from the upstream GLM fork plus the NVFP4 / 8-rank changes described in
   [`docs/design.md`](docs/design.md). (Not published here.)
4. **Launch:** 8 local ranks, one GPU each, NUMA-pinned; `--tp 8 --context 262144 --parallel 32
   --mtp-drafts 2`; NCCL on loopback with 16 channels; the engine's own OpenAI-compatible server on a local
   port (examples use `8000`).
5. **Serving settings that matter** (with their measured effect): CUDA-IPC decode gather, L2 prefetch, the
   exact wide reduce, a 432-row batched fill, a host-RAM prompt tier (40 GiB/rank = 320 GiB pinned),
   fill-in-slices fairness, prompt reuse, growing reply room, streamed tool calls. See the recipe for the
   flag table and warm-up/health checks.

## Benchmarks

Tables, dates, methods and the exact settings for each run — including the worse results and a like-for-like
comparison with a public vLLM-based setup — are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Design notes

What we changed on top of the GLM fork and why (prose, no code): [`docs/design.md`](docs/design.md).

## Credits

- **TensorFold** — <https://tensorfold.dev>.
- **`bertholomus/TensorFold`, branch `glm-dsa-tp4`** (Apache-2.0) — the Python TensorFold fork with the
  full GLM-5.3 (`glm_moe_dsa`) CUDA family this work builds on.
- **`zai-org/GLM-5.3`** — the base model; **`dealignai/GLM-5.3-UNCENSORED-NVFP4`** — the served checkpoint
  (MIT).
- **`kacper-daftcode/GLM5.3-8xPRO6K`** (Apache-2.0) — a public, patched-vLLM serving stack for the same
  class of machine, used for the like-for-like comparison in the benchmarks.
- **`kindlingai/gr8`** (MIT) — the gr8bench method, prompts and checkers adapted for one benchmark run
  (credited in place in `bench/`).

## License

Apache-2.0 for the documentation and the scripts in this repository — see [`LICENSE`](LICENSE).
© the authors. The model and the upstream engine carry their own licenses (MIT and Apache-2.0,
respectively).

## Honest limits

- **The engine changes are not public yet.** This repo is a recipe plus numbers; it does not let you build
  the serving engine from source.
- **All numbers come from one machine**, measured over 2026-11-13…15. They are not a cross-hardware study,
  and run-to-run spread below ~2 % is noise (see the benchmark caveats).
- An **FP4-experts (W4A4) experiment** that would speed up prefill was lossy (it changes replies and dropped
  GSM8K from 97.19 % to 96.89 %) and is **parked, off by default** — the served path is exact.
- Some things the public comparison measured, we did not (e.g. a multi-hour mixed soak, a 500k+ needle;
  our context is 262,144 tokens). These gaps are listed in the benchmarks doc.
