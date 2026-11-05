#!/usr/bin/env python3
"""Extract load timings from the pod logs of scripts/torchtpu_load_probe.py runs into summary.json."""
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])  # results/torchtpu-load/pod_logs/logs
out = {}
TS = re.compile(r'^(\d\d:\d\d:\d\d\.\d+) ')


def secs(t):
    return datetime.strptime(t, '%H:%M:%S.%f').timestamp()


for run in sorted(p for p in root.iterdir() if p.is_dir()):
    log = run / 'vllm_raw.log'
    if not log.exists():
        continue
    r = {'workers': {}}
    lines = log.read_text(errors='replace').splitlines()
    r['log_lines'] = len(lines)
    for ln in lines:
        m = TS.match(ln)
        if not m:
            continue
        t = m.group(1)
        w = re.search(r'\(Worker_TP(\d) pid=(\d+)\)', ln)
        wk = r['workers'].setdefault('TP' + w.group(1), {}) if w else None
        if 'Loading model from scratch' in ln and wk is not None:
            wk['load_start'] = t
        if 'Loading weights took' in ln:
            r.setdefault('vllm_native_loading_weights_took_s', []).append(
                {'worker': 'TP' + w.group(1) if w else None, 'seconds': float(re.search(r'took ([\d.]+) seconds', ln).group(1)), 'at': t})
        if 'TTL_PROBE load_weights' in ln and wk is not None:
            d = dict(re.findall(r'(\w[\w+]*)=([^\s]+)', ln.split('TTL_PROBE load_weights')[1]))
            wk['probe_load_weights'] = d
            wk['load_weights_end'] = t
        if 'TTL_PROBE load_model' in ln and wk is not None:
            d = dict(re.findall(r'(\w[\w+]*)=([^\s]+)', ln.split('TTL_PROBE load_model')[1]))
            wk['probe_load_model'] = d
            wk['load_model_end'] = t
        if 'Memory statistics' in ln and wk is not None and 'memory_statistics' not in wk:
            wk['memory_statistics'] = ln.split('Memory statistics | ')[1]
        if 'GPU KV cache size' in ln:
            r['kv_cache'] = ln.split('] ', 1)[1]
        if 'init engine (profile, create kv cache, warmup model) took' in ln:
            r['init_engine'] = ln.split('] ', 1)[1]
        if 'Application startup complete' in ln:
            r['ready_at'] = t
        if 'EngineCore failed to start' in ln and 'engine_failed_at' not in r:
            r['engine_failed_at'] = t
        if re.search(r'ValueError: .*(exceeds|KV cache)', ln) and 'failure' not in r:
            r['failure'] = ln.split('] ', 1)[-1][:400]
        if 'non-default args' in ln and 'args' not in r:
            r['args'] = ln.split('non-default args: ')[1]
        if re.search(r'RunAI Streamer|runai|Run:AI|Runai', ln) and len(r.setdefault('runai_lines', [])) < 40:
            r['runai_lines'].append(ln[:300])
        if 'torch.compile took' in ln:
            r.setdefault('torch_compile_took', []).append(ln.split('] ', 1)[-1].strip())
        if 'Worker_TP0' in ln and re.search(r'Compilation finished in [\d.]+ \[secs\]', ln):
            r.setdefault('tp0_compilation_finished_s', []).append(float(re.search(r'in ([\d.]+) \[secs\]', ln).group(1)))
    for wk in r['workers'].values():
        if 'load_start' in wk and 'load_model_end' in wk:
            wk['wall_load_start_to_load_model_end_s'] = round(secs(wk['load_model_end']) - secs(wk['load_start']), 2)
    host = (run / 'host.log').read_text().splitlines() if (run / 'host.log').exists() else []
    for ln in host:
        if 'vllm launched' in ln:
            r['launched_at'] = ln.split()[0]
        if 'PAGECACHE_JSON=' in ln:
            r.setdefault('page_cache', []).append(json.loads(ln.split('PAGECACHE_JSON=', 1)[1]))
    net = run / 'net.tsv'
    if net.exists():
        rows = [tuple(map(float, x.split('\t'))) for x in net.read_text().splitlines()[1:] if x.count('\t') == 2]
        day = datetime.fromtimestamp(rows[0][0], timezone.utc).strftime('%Y-%m-%d')

        def epoch(hms):
            return datetime.strptime(f'{day} {hms}', '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()

        def rx_at(t):  # last sample at or before t
            best = rows[0]
            for row in rows:
                if row[0] <= t:
                    best = row
            return best

        launch = epoch(r['launched_at']) if 'launched_at' in r else rows[0][0]
        b0, b1 = rows[0], rx_at(launch)
        n = {'samples': len(rows), 'span_s': round(rows[-1][0] - rows[0][0], 1),
             'rx_total_GB': round((rows[-1][1] - rows[0][1]) / 1e9, 2),
             'baseline_rx_MBps': round((b1[1] - b0[1]) / max(1e-9, b1[0] - b0[0]) / 1e6, 2) if b1[0] > b0[0] else None}
        starts = [float(w['probe_load_weights']['wall_start']) for w in r['workers'].values() if 'probe_load_weights' in w]
        ends = [s + float(w['probe_load_weights']['call'].rstrip('s')) for s, w in
                zip(starts, [w for w in r['workers'].values() if 'probe_load_weights' in w])]
        if starts:
            lo, hi = rx_at(min(starts)), rx_at(max(ends) + 0.5)
            n['load_window_s'] = round(max(ends) - min(starts), 2)
            n['rx_load_window_GB'] = round((hi[1] - lo[1]) / 1e9, 2)
            n['rx_load_window_GBps'] = round((hi[1] - lo[1]) / max(1e-9, hi[0] - lo[0]) / 1e9, 3)
        peak = 0.0
        for i, row in enumerate(rows):
            j = next((k for k in range(i, len(rows)) if rows[k][0] - row[0] >= 5.0), None)
            if j is not None:
                peak = max(peak, (rows[j][1] - row[1]) / (rows[j][0] - row[0]))
        n['peak_5s_rx_GBps'] = round(peak / 1e9, 3)
        r['net'] = n
    g = run / 'greedy_1.json'
    if g.exists():
        j = json.loads(g.read_text())
        r['greedy_response'] = j['choices'][0]['message']['content']
        r['greedy_finish'] = j['choices'][0]['finish_reason']
    sr = run / 'serve_result.json'
    if sr.exists():
        r['ready'] = json.loads(sr.read_text())['ready']
    if 'launched_at' in r and 'ready_at' in r:
        r['launch_to_ready_s'] = round(secs(r['ready_at']) - secs(r['launched_at']), 1)
    rp = run / 'req_probe.json'
    if rp.exists():
        r['req_probe'] = [{k: v for k, v in q.items() if k != 'text'} | {'text': q.get('text', '')[:120]}
                          for q in json.loads(rp.read_text())['requests']]
        r['greedy_response'] = r['req_probe'][0].get('text')
    out[run.name] = r
(root.parent / 'summary_extract.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
