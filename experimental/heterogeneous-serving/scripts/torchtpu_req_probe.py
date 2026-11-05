#!/usr/bin/env python3
"""Timed first requests against a just-ready vLLM server (stdlib only; runs on the TPU host with the image's python3.12).

Streams each chat completion and records time to first token, total time, completion tokens and the decode rate
after the first token. The sequence exposes lazy work left for serving time (an eager engine compiles per new shape):
  q1 short greedy prompt (cold), q1 again (warm), q2 ~300-token prompt (new prefill shape), q2 again,
  d128: 128 forced decode tokens (ignore_eos) for a steady decode rate.
Usage: torchtpu_req_probe.py PORT OUT.json
"""
import json
import sys
import time
import urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
SHORT = 'What is the capital of France? Answer in one sentence.'
PARA = ('The Tensor Processing Unit is an application-specific integrated circuit developed for neural network '
        'machine learning. Chips are connected by an inter-chip interconnect into slices, and a slice runs one '
        'program across all of its chips. ')
LONG = PARA * 8 + 'Summarize the text above in two sentences.'


def chat(prompt, max_tokens, ignore_eos=False):
    body = {'model': 'gemma4', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': max_tokens, 'stream': True, 'stream_options': {'include_usage': True}}
    if ignore_eos:
        body['ignore_eos'] = True
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = last = None
    chunks, text, usage = 0, [], None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            d = json.loads(line[6:])
            if d.get('usage'):
                usage = d['usage']
            for c in d.get('choices', []):
                piece = c.get('delta', {}).get('content')
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
                    text.append(piece)
    t1 = time.perf_counter()
    n = (usage or {}).get('completion_tokens')
    return {'ttft_s': round(first - t0, 3) if first else None, 'total_s': round(t1 - t0, 3),
            'prompt_tokens': (usage or {}).get('prompt_tokens'), 'completion_tokens': n,
            'decode_tok_per_s_after_first': round((n - 1) / (last - first), 2) if n and n > 1 and last > first else None,
            'text': ''.join(text)}


plan = [('q1_cold', SHORT, 32, False), ('q1_warm', SHORT, 32, False), ('q2_newshape', LONG, 64, False),
        ('q2_warm', LONG, 64, False), ('d128', SHORT, 128, True)]
out = {'wall_start': time.time(), 'requests': []}
for name, prompt, mt, ie in plan:
    try:
        r = chat(prompt, mt, ie)
    except Exception as e:  # noqa: BLE001
        r = {'error': repr(e)}
    r['name'] = name
    out['requests'].append(r)
    print('REQ_PROBE_JSON=' + json.dumps({k: v for k, v in r.items() if k != 'text'} | {'text': r.get('text', '')[:80]}),
          flush=True)
with open(OUT, 'w') as f:
    json.dump(out, f, indent=1)
