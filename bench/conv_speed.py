#!/usr/bin/env python3
"""A 4-turn conversation against an OpenAI-compatible server, streamed, with per-chunk timing:
how fast a reply starts, runs in the middle and ends, turn by turn as the context grows.

  python3 -I conv_speed.py OUT_DIR [--url http://127.0.0.1:8000] [--model MODEL]
                           [--tokenizer tokenizer.json] [--effort low|high]

Tokens a chunk are counted with the model's own tokenizer (a chunk = what one engine round emitted for this stream);
the per-turn total is checked against usage.completion_tokens. /health is sampled every 0.5 s to see other users.
Needs the `tokenizers` package for the tokenizer; everything else is the standard library.
"""
import argparse, json, os, threading, time, urllib.request

TURNS = [
    'Write a calculator in Python.',
    'Now explain every line of that code: what it is for and how it runs.',
    'Modify the calculator so it can also be opened and used through a web browser.',
    'Now explain every line of the new code again: what it is for and how it runs.',
]
MAX_TOKENS = 32000

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="", help="default: the first id /v1/models lists")
ap.add_argument("--tokenizer", required=True, help="path to the model's tokenizer.json")
ap.add_argument("--effort", default=None, help="reasoning_effort for every turn (default: the server's own)")
args = ap.parse_args()

from tokenizers import Tokenizer
tok = Tokenizer.from_file(args.tokenizer)
URL = args.url.rstrip("/")
out = args.out; os.makedirs(out, exist_ok=True)
EFFORT = args.effort
with urllib.request.urlopen(URL + '/v1/models', timeout=30) as r:
    MODEL = args.model or json.load(r)["data"][0]["id"]
health, stop = [], threading.Event()


def sample_health():
    while not stop.is_set():
        try:
            h = json.load(urllib.request.urlopen(URL + '/health', timeout=2))
            health.append((time.time(), h.get('requests_running'), h.get('streams', {}).get('decoding')))
        except Exception:
            pass
        time.sleep(0.5)


def turn(messages, i):
    body = {'model': MODEL, 'messages': messages, 'max_tokens': MAX_TOKENS, 'stream': True,
            'stream_options': {'include_usage': True}}
    if EFFORT: body['reasoning_effort'] = EFFORT
    req = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time(); chunks = []; content = []; reasoning = []; final = {}
    with urllib.request.urlopen(req, timeout=3600) as f:
        for raw in f:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            j = json.loads(line[5:])
            now = time.time()
            if j.get('usage'): final['usage'] = j['usage']
            if j.get('tensorfold'): final['tensorfold'] = j['tensorfold']
            for c in j.get('choices', []):
                d = c.get('delta', {})
                r, t = d.get('reasoning_content') or '', d.get('content') or ''
                if r or t:
                    n = len(tok.encode(r + t, add_special_tokens=False).ids)
                    chunks.append({'t': now - t0, 'n': n, 'kind': 'reasoning' if r else 'content'})
                    reasoning.append(r); content.append(t)
                if c.get('finish_reason'): final['finish'] = c['finish_reason']
    return {'turn': i + 1, 'chunks': chunks, 'content': ''.join(content), 'reasoning': ''.join(reasoning), **final,
            'wall_s': time.time() - t0}


def windows(chunks):
    """tok/s in the first, middle and last tenth of the reply's tokens, and in 10-second bins."""
    total = sum(c['n'] for c in chunks)
    if total < 20: return {}
    cum, out = 0, []
    for c in chunks:
        cum += c['n']; out.append((c['t'], cum))
    def rate(lo, hi):
        pts = [(t, k) for t, k in out if lo * total <= k <= hi * total]
        if len(pts) < 2: return None
        return round((pts[-1][1] - pts[0][1]) / max(1e-6, pts[-1][0] - pts[0][0]), 1)
    bins, b0, k0 = [], out[0][0], out[0][1]
    for t, k in out:
        if t - b0 >= 10:
            bins.append(round((k - k0) / (t - b0), 1)); b0, k0 = t, k
    return {'tokens_counted': total, 'first_token_s': round(chunks[0]['t'], 3), 'start_tps': rate(0.0, 0.1),
            'middle_tps': rate(0.45, 0.55), 'end_tps': rate(0.9, 1.0), 'overall_tps': rate(0.0, 1.0), 'bins_10s': bins}


th = threading.Thread(target=sample_health, daemon=True); th.start()
msgs, results = [], []
for i, q in enumerate(TURNS):
    msgs.append({'role': 'user', 'content': q})
    t_start = time.time()
    r = turn(msgs, i)
    r['window'] = windows(r['chunks'])
    hs = [h for h in health if t_start <= h[0] <= t_start + r['wall_s']]
    r['others_running_max'] = max([(h[1] or 0) - 1 for h in hs], default=None)
    msgs.append({'role': 'assistant', 'content': r['content']})
    results.append(r)
    w, tf, u = r['window'], r.get('tensorfold', {}), r.get('usage', {})
    print(json.dumps({'turn': i + 1, 'prompt_tokens': u.get('prompt_tokens'), 'cached': (u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                      'completion_tokens': u.get('completion_tokens'), 'reasoning_chars': len(r['reasoning']), 'content_chars': len(r['content']),
                      'finish': r.get('finish'), **{k: w.get(k) for k in ('tokens_counted', 'first_token_s', 'start_tps', 'middle_tps', 'end_tps', 'overall_tps')},
                      'bins_10s': w.get('bins_10s'), 'rounds': tf.get('rounds'), 'drafted': tf.get('drafted'), 'accepted': tf.get('accepted'),
                      'round_ms': tf.get('round_ms'), 'others_running_max': r['others_running_max']}), flush=True)
stop.set()
json.dump(results, open(os.path.join(out, 'conv.json'), 'w'), indent=1)
json.dump(health, open(os.path.join(out, 'health.json'), 'w'))
