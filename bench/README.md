# bench/ — client-side benchmark and test scripts

Standard-library Python 3 (3.12 was used). Each script talks HTTP to **any OpenAI-compatible endpoint** —
it only sends requests to the `--url` you give and reads the replies. Nothing here imports an engine or
touches a GPU. The default endpoint is `http://127.0.0.1:8000`; pass `--url` for anything else.

The model name is taken from the server's `/v1/models` unless you pass `--model`. Most scripts print one
machine-readable summary line and, with `--log FILE`, append it (with a UTC timestamp) to a log.

Some scripts read extra fields the server may put in each reply (`tensorfold` round stats, `usage`
cached-token details). Against a server that doesn't return them, the timing figures still work; the
per-round breakdowns are just blank.

## The scripts

| Script | What it measures | Typical call |
|---|---|---|
| `speed_probe.py` | One-request decode speed and where each round's time goes; checks drafted = serial | `python3 speed_probe.py --url URL --tokens 384 --label p` |
| `speedtest.py` | Total tok/s and time-to-first-token at 1..N parallel requests, thinking on/off | `python3 -I speedtest.py --url URL --levels 1,2,4,8,16,32 --tokens 256` |
| `wide_probe.py` | N requests together with each reply's own round stats (rows, stages) | `python3 wide_probe.py --url URL --streams 32 --tokens 512` |
| `together_check.py` | **Exactness:** N prompts one at a time, then all at once; every reply must match | `python3 together_check.py --url URL --prompts-file FILE --streams 32 --label t` |
| `longctx.py` | Prefill time and post-prefill decode tok/s for a long prompt; optional extend/N-streams | `python3 longctx.py --url URL --tokens 72000 --label lc` |
| `reuse_check.py` | Prompt reuse: what a kept prompt saves on a growing conversation; resumed = fresh | `python3 reuse_check.py --url URL --tokens 16000 --grow 2000 --steps 3 --label r` |
| `session_probe.py` | Two large sessions whose prefixes evict each other (host-RAM tier restore) | `python3 session_probe.py --url URL --tokens 140000 --steps 4 --label s` |
| `fair_probe.py` | What a long prompt's read does to decoders and a short request beside it | `python3 fair_probe.py --url URL --long 131072 --others 4 --label f` |
| `grow_probe.py` | Growing reply room (a huge request beside short ones) and streamed tool-call deltas | `python3 grow_probe.py long --url URL --long 200000 --label g` |
| `conv_speed.py` | A 4-turn coding conversation with per-chunk timing as context grows | `python3 -I conv_speed.py OUT_DIR --url URL --tokenizer tokenizer.json` |

Run `python3 SCRIPT.py --help` (or read the module docstring) for every option and the exact format of the
summary line.

### Dependencies

- All scripts: Python 3 standard library only.
- `conv_speed.py` additionally needs the `tokenizers` package (`pip install tokenizers==0.23.3` — pin the
  version you benchmark with, so a later release cannot silently change what you measure) and the model's
  `tokenizer.json` (pass `--tokenizer PATH`) to count tokens per streamed chunk.
- `session_probe.py`, `fair_probe.py` and `grow_probe.py` import helpers from `longctx.py` (and
  `reuse_check.py`); keep the files together in this directory.

### Notes on method

- "Together" totals are all tokens of a level over the wall-clock time from the first send to the last
  reply. `speedtest.py` and `wide_probe.py` use distinct prose/code questions with a unique first line so
  no request reuses another's prompt cache; `together_check.py` uses one prompts file you supply.
- Thinking is controlled with `chat_template_kwargs.enable_thinking` (off) or the server's default (on),
  per script.
- For a like-for-like concurrency number comparable across engines, use 256-token replies with
  `ignore_eos` and temperature 0 (the mode `together_check.py` and the decode-style runs use).

## Credit

`conv_speed.py`, the probes and the exactness check are original to this project (Apache-2.0, © the
authors). The gr8bench-style run referenced in [`../docs/benchmarks.md`](../docs/benchmarks.md) §7 used
method, prompt texts, task sets and checkers from **gr8bench** (`bench/gr8bench` of
<https://github.com/kindlingai/gr8>, MIT License); that harness is not included here.


Note: the non-English prompts in `speed_probe.py` / `speedtest.py` are Spanish here; the measured runs used a
different non-English language, so per-prompt numbers can differ by a few percent.
