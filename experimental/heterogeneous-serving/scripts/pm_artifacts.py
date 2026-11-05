#!/usr/bin/env python3
"""TPUFlex startup state on GCS for fresh-host (autoscale) boots of vLLM-TorchTPU.

  pack    --model-dir D --bank-dir B --xla-dir X --snap-dir S --out O
          model.tar  = the checkpoint dir without weights (config, tokenizer, chat template, index; ~32 MB)
          state.tar  = bank/ (VLLM_CACHE_ROOT: executable bank, AOT forwards, model info) + torch_tpu_cache/
          files.json = object list: model.tar, state.tar, snap/rank<R>.bin, snap/rank<R>.json (local paths, sizes)
  upload  --pack O --dst gs://bucket/prefix [--workers N]     (needs google-cloud-storage: the venv's python)
          big files with transfer_manager.upload_chunks_concurrently (XML multipart), files.json last
  fetch   --src gs://bucket/prefix --lib libstreamer.so --model-dir D --bank-dir B --xla-dir X --snap-dir S
          stdlib + the Run:ai streamer's C library through ctypes (no torch import). files.json over the JSON
          API, then model.tar (extracted, then <snap-dir>/../.model_done), then one streamer request for
          state.tar and every rank file, written straight into mmap'd tmpfs files. On completion: state.tar is
          extracted (then .state_done); rank<R>.json is written only after rank<R>.bin is complete, so it is
          the commit marker the plugin's restore waits for. Prints FETCH_JSON={...} with per-file timings.
  delete  --dst gs://bucket/prefix                             (venv python) remove every object under prefix

Objects are stored under <prefix>/<name> with name as in files.json.
"""
import argparse
import ctypes
import io
import json
import mmap
import os
import shutil
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request

RANGE = 256 << 20
MD = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/'


def split_gs(uri):
    bucket, _, prefix = uri.removeprefix('gs://').partition('/')
    return bucket, prefix.strip('/')


# ------------------------------------------------------------------------------------------------ pack

def cmd_pack(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    files = []
    model_tar = os.path.join(a.out, 'model.tar')
    with tarfile.open(model_tar, 'w') as tf:
        for name in sorted(os.listdir(a.model_dir)):
            p = os.path.join(a.model_dir, name)
            if os.path.isfile(p) and not name.endswith('.safetensors') and not name.startswith('.'):
                tf.add(p, arcname=name)
    files.append({'name': 'model.tar', 'path': model_tar, 'role': 'model'})
    state_tar = os.path.join(a.out, 'state.tar')
    with tarfile.open(state_tar, 'w') as tf:
        tf.add(a.bank_dir, arcname='bank')
        if os.path.isdir(a.xla_dir):
            tf.add(a.xla_dir, arcname='torch_tpu_cache')
    files.append({'name': 'state.tar', 'path': state_tar, 'role': 'state'})
    ranks = sorted(int(n[4:-5]) for n in os.listdir(a.snap_dir) if n.startswith('rank') and n.endswith('.json'))
    for r in ranks:
        for ext in ('bin', 'json'):
            p = os.path.join(a.snap_dir, f'rank{r}.{ext}')
            files.append({'name': f'snap/rank{r}.{ext}', 'path': p, 'role': f'snap_{ext}', 'rank': r})
    for f in files:
        f['size'] = os.path.getsize(f['path'])
    with open(os.path.join(a.out, 'files.json'), 'w') as fh:
        json.dump({'files': files}, fh, indent=1)
    print('PACK_JSON=' + json.dumps({'seconds': round(time.time() - t0, 2), 'files': [
        {k: f[k] for k in ('name', 'size')} for f in files]}), flush=True)


# ---------------------------------------------------------------------------------------------- upload

def cmd_upload(a):
    from google.cloud import storage
    from google.cloud.storage import transfer_manager as tm
    bucket_name, prefix = split_gs(a.dst)
    bucket = storage.Client().bucket(bucket_name)
    with open(os.path.join(a.pack, 'files.json')) as fh:
        man = json.load(fh)
    t0 = time.time()
    total = 0
    for f in man['files']:
        t1 = time.time()
        blob = bucket.blob(f'{prefix}/{f["name"]}')
        if f['size'] > RANGE:
            tm.upload_chunks_concurrently(f['path'], blob, chunk_size=128 << 20, max_workers=a.workers,
                                          worker_type=tm.PROCESS)
        else:
            blob.upload_from_filename(f['path'])
        total += f['size']
        print(f'uploaded {f["name"]} {f["size"] / 1e9:.2f} GB in {time.time() - t1:.1f} s', flush=True)
    public = {'files': [{k: v for k, v in f.items() if k != 'path'} for f in man['files']]}
    bucket.blob(f'{prefix}/files.json').upload_from_string(json.dumps(public), content_type='application/json')
    dt = time.time() - t0
    print('UPLOAD_JSON=' + json.dumps({'dst': a.dst, 'bytes': total, 'seconds': round(dt, 1),
                                       'GBps': round(total / dt / 1e9, 2)}), flush=True)


def cmd_delete(a):
    from google.cloud import storage
    bucket_name, prefix = split_gs(a.dst)
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix + '/'))
    for b in blobs:
        b.delete()
    print('DELETE_JSON=' + json.dumps({'dst': a.dst, 'objects': len(blobs),
                                       'bytes': sum(b.size or 0 for b in blobs)}), flush=True)


# ----------------------------------------------------------------------------------------------- fetch

class Streamer:
    """The Run:ai streamer C API (runai_start / runai_request / runai_response / runai_end) via ctypes."""

    def __init__(self, lib_path):
        L = ctypes.CDLL(lib_path)
        vp, sz, u32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32
        L.runai_start.argtypes, L.runai_start.restype = [ctypes.POINTER(vp)], ctypes.c_int
        L.runai_end.argtypes = [vp]
        L.runai_request.argtypes = [vp, u32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(sz),
                                    ctypes.POINTER(sz), ctypes.POINTER(vp), ctypes.POINTER(u32),
                                    ctypes.POINTER(ctypes.POINTER(sz))] + [ctypes.c_char_p] * 5
        L.runai_request.restype = ctypes.c_int
        L.runai_response.argtypes = [vp, ctypes.POINTER(u32), ctypes.POINTER(u32)]
        L.runai_response.restype = ctypes.c_int
        L.runai_response_str.argtypes, L.runai_response_str.restype = [ctypes.c_int], ctypes.c_char_p
        self.L = L
        self.h = vp(0)
        self._check(L.runai_start(ctypes.byref(self.h)), 'start')

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError(f'runai_{what}: {self.L.runai_response_str(rc).decode()}')

    def request(self, paths, sizes, dsts, ranges, offsets=None):
        """paths[i] (from offsets[i], default 0; sizes[i] bytes) -> writable buffer dsts[i], completed in
        ranges[i] pieces."""
        n = len(paths)
        keep = [ctypes.c_char.from_buffer(d) for d in dsts]
        c_paths = (ctypes.c_char_p * n)(*[p.encode() for p in paths])
        c_offs = (ctypes.c_size_t * n)(*(offsets or [0] * n))
        c_sizes = (ctypes.c_size_t * n)(*sizes)
        c_dsts = (ctypes.c_void_p * n)(*[ctypes.addressof(k) for k in keep])
        c_nr = (ctypes.c_uint32 * n)(*[len(r) for r in ranges])
        arrs = [(ctypes.c_size_t * len(r))(*r) for r in ranges]
        c_ranges = (ctypes.POINTER(ctypes.c_size_t) * n)(
            *[ctypes.cast(x, ctypes.POINTER(ctypes.c_size_t)) for x in arrs])
        self._keep = (keep, c_paths, c_offs, c_sizes, c_dsts, c_nr, arrs, c_ranges)
        self._check(self.L.runai_request(self.h, n, c_paths, c_offs, c_sizes, c_dsts, c_nr, c_ranges,
                                         None, None, None, None, None), 'request')

    def responses(self, n):
        """Wait for exactly n range completions, as runai_model_streamer's FileStreamer does. Looping until
        FINISHED is wrong: on the second request of a handle FINISHED came back before the bytes had landed
        (a 149 KB rank manifest read back as zeros)."""
        fi, ri = ctypes.c_uint32(), ctypes.c_uint32()
        for k in range(n):
            rc = self.L.runai_response(self.h, ctypes.byref(fi), ctypes.byref(ri))
            if rc == 1:
                raise RuntimeError(f'runai_response: FINISHED after {k} of {n} ranges')
            self._check(rc, 'response')
            yield fi.value, ri.value

    def release(self):
        self._keep = None

    def close(self):
        self.L.runai_end(self.h)


def split_ranges(size):
    if size == 0:
        return [0]
    return [min(RANGE, size - o) for o in range(0, size, RANGE)]


def gcs_token():
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        MD + 'token', headers={'Metadata-Flavor': 'Google'}), timeout=10).read())['access_token']


def gcs_bytes(bucket, name):
    url = (f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/'
           f'{urllib.parse.quote(name, safe="")}?alt=media')
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {gcs_token()}'})
    return urllib.request.urlopen(req, timeout=30).read()


def gcs_list(bucket, prefix):
    """Objects directly under prefix/ as [(name relative to prefix, size)] (JSON API, all pages)."""
    auth = {'Authorization': f'Bearer {gcs_token()}'}
    out, page = [], None
    while True:
        q = {'prefix': prefix + '/', 'delimiter': '/', 'fields': 'items(name,size),nextPageToken'}
        if page:
            q['pageToken'] = page
        url = f'https://storage.googleapis.com/storage/v1/b/{bucket}/o?' + urllib.parse.urlencode(q)
        r = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=auth), timeout=30).read())
        out += [(it['name'][len(prefix) + 1:], int(it['size'])) for it in r.get('items', [])
                if it['name'] != prefix + '/']  # the console's "folder" placeholder object
        page = r.get('nextPageToken')
        if not page:
            return out


def gcs_json(bucket, name):
    return json.loads(gcs_bytes(bucket, name))


class _View(io.RawIOBase):
    """Seekable read-only file over a buffer, without copying it (io.BytesIO copies a bytearray)."""

    def __init__(self, buf):
        self.mv, self.pos = memoryview(buf), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def readinto(self, b):
        k = max(0, min(len(b), len(self.mv) - self.pos))
        b[:k] = self.mv[self.pos:self.pos + k]
        self.pos += k
        return k

    def seek(self, off, whence=0):
        self.pos = (off, self.pos + off, len(self.mv) + off)[whence]
        return self.pos

    def tell(self):
        return self.pos


def extract(buf, dest_map, threads=16):
    """Extract tar bytes; members whose first path component is in dest_map go to that directory.
    Returns the number of members extracted (0 for an all-zero buffer, which tarfile reads as an empty archive).
    Regular files are written straight from the buffer, one os.write per file, on `threads` threads: tarfile's
    own 16 KB copy loop took 3-19 s for the 2.8-3.9 GB state.tar (f1, f6-f9, f16, f17)."""
    from concurrent.futures import ThreadPoolExecutor
    files, n = [], 0
    with tarfile.open(fileobj=_View(buf), mode='r:') as tf:
        for m in tf.getmembers():
            top, _, rest = m.name.partition('/')
            base = dest_map.get(top) if top in dest_map else dest_map.get('')
            if base is None:
                continue
            m.name = rest if top in dest_map else m.name
            if not m.name:
                continue
            if m.isreg():
                safe = tarfile.data_filter(m, base)  # same path checks as filter='data'
                files.append((os.path.join(base, safe.name), m.offset_data, m.size))
            else:
                tf.extract(m, base, filter='data')
            n += 1
    mv = memoryview(buf)

    def write(job):
        path, off, size = job
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            view = mv[off:off + size]
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(write, files))
    return n


def stream_into(st, lib, url, size, dst, base=0, spare=None, tag=''):
    """url[base:base+size] -> dst (a writable buffer of `size` bytes) in RANGE pieces on streamer st, one file per
    request. libstreamer sometimes returns FINISHED before every range completed, on gs:// at random (f16: three
    of four ranged requests; f20: one of four rank snapshots, and the masked error left that rank waiting): the
    acknowledged ranges are kept and the span of the missing ones is re-requested on a new streamer, appended to
    `spare`, which the caller closes at the end (it may still reference dst). Up to 4 attempts.
    Returns True if the first attempt completed (st released, dst no longer referenced)."""
    ranges = split_ranges(size)
    todo = list(range(len(ranges)))
    for attempt in range(4):
        lo, hi = todo[0], todo[-1]
        sub = ranges[lo:hi + 1]
        o = lo * RANGE
        if attempt:
            st = Streamer(lib)
            spare.append(st)
        st.request([url], [sum(sub)], [memoryview(dst)[o:o + sum(sub)]], [sub], offsets=[base + o])
        done = set()
        try:
            for _, ri in st.responses(len(sub)):
                done.add(lo + ri)
        except RuntimeError as e:
            if 'FINISHED' not in str(e) or attempt == 3 or spare is None:
                raise
            print(f'stream {tag}@{base} attempt {attempt}: {e}; re-requesting the missing ranges', flush=True)
        todo = [i for i in todo if i not in done]
        if not todo:
            if attempt == 0:
                st.release()
            return attempt == 0


def cmd_fetch(a):
    T0 = time.time()
    log = {}
    stamp = lambda k: log.__setitem__(k, round(time.time() - T0, 3))  # noqa: E731
    if a.src.startswith('gs://'):
        bucket, prefix = split_gs(a.src)
        man = gcs_json(bucket, f'{prefix}/files.json')['files']
        url = lambda name: f'gs://{bucket}/{prefix}/{name}'  # noqa: E731
        small = lambda name: gcs_bytes(bucket, f'{prefix}/{name}')  # noqa: E731
    else:  # a local directory laid out like the bucket prefix (tests)
        with open(os.path.join(a.src, 'files.json')) as fh:
            man = json.load(fh)['files']
        url = lambda name: os.path.join(a.src, name)  # noqa: E731
        small = lambda name: open(os.path.join(a.src, name), 'rb').read()  # noqa: E731, SIM115
    stamp('manifest')
    by_role = {}
    for f in man:
        by_role.setdefault(f['role'], []).append(f)
    os.makedirs(a.snap_dir, exist_ok=True)
    root = os.path.dirname(os.path.abspath(a.snap_dir))
    errors, spare, pinned = [], [], []

    def stream_one(st, f, dst):
        """One file per request: with several files in one request libstreamer wrote file i at
        dsts[0] + (bytes of files before i), i.e. it assumes one contiguous destination (EFAULT otherwise)."""
        clean = stream_into(st, a.lib, url(f['name']), f['size'], dst, spare=spare, tag=f['name'])
        stamp(f['name'] + '_done')
        return clean

    def stream_to_file(st, f, path):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, f['size'])
        mm = mmap.mmap(fd, f['size'])
        os.close(fd)
        pinned.append(mm)  # closed here once nothing references it, else at exit
        if stream_one(st, f, mm):
            pinned.remove(mm)
            mm.close()

    def guarded(fn):
        def run():
            try:
                fn()
            except BaseException as e:  # noqa: BLE001
                errors.append(repr(e))
                print(f'fetch: ERROR {e!r}', flush=True)
        return run

    # 1) the weight-less model dir first: vLLM's API server reads it within seconds of launch
    st = Streamer(a.lib)
    stamp('streamer_up')
    if not a.state_only:
        mf = by_role['model'][0]
        buf = bytearray(mf['size'])
        stream_one(st, mf, buf)
        os.makedirs(a.model_dir, exist_ok=True)
        if not extract(buf, {'': a.model_dir}) or not os.path.exists(os.path.join(a.model_dir, 'config.json')):
            raise RuntimeError('model.tar: no config.json extracted')
        open(os.path.join(root, '.model_done'), 'w').close()
        stamp('model_done')
        print(f'fetch: model dir ready {log["model_done"]:.2f} s', flush=True)

    # 2) startup state and each rank's terminal snapshot, one streamer (own connections) per thread
    def state_job():
        sf = by_role['state'][0]
        sbuf = bytearray(sf['size'])
        stream_one(st, sf, sbuf)
        if not extract(sbuf, {'bank': a.bank_dir, 'torch_tpu_cache': a.xla_dir}):
            raise RuntimeError('state.tar: nothing extracted')
        open(os.path.join(root, '.state_done'), 'w').close()
        stamp('state_done')
        print(f'fetch: startup state extracted {log["state_done"]:.2f} s', flush=True)

    def rank_job(fb, fj):
        rst = Streamer(a.lib)
        try:
            stream_to_file(rst, fb, os.path.join(a.snap_dir, os.path.basename(fb['name'])))
        finally:
            rst.close()
        jbuf = small(fj['name'])  # 149 KB: one plain GET, then parse before it becomes the commit marker
        if len(jbuf) != fj['size']:
            raise RuntimeError(f'{fj["name"]}: {len(jbuf)} bytes, expected {fj["size"]}')
        json.loads(jbuf)
        tmp = os.path.join(a.snap_dir, os.path.basename(fj['name']) + '.tmp')
        with open(tmp, 'wb') as fh:
            fh.write(jbuf)
        os.replace(tmp, tmp[:-4])  # rank<R>.json last: the restore's commit marker
        stamp(f'rank{fb["rank"]}_committed')
        print(f'fetch: rank{fb["rank"]} committed {log[f"rank{fb["rank"]}_committed"]:.2f} s', flush=True)

    jsons = {f['rank']: f for f in by_role.get('snap_json', [])}
    threads = [threading.Thread(target=guarded(state_job))]
    threads += [threading.Thread(target=guarded(lambda fb=fb: rank_job(fb, jsons[fb['rank']])))
                for fb in ([] if a.state_only else by_role.get('snap_bin', []))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st.close()
    for s in spare:
        s.close()
    stamp('all_done')
    total = sum(f['size'] for f in man if not a.state_only or f['role'] == 'state')
    print('FETCH_JSON=' + json.dumps({'src': a.src, 'bytes': total, 'seconds': log['all_done'], 'errors': errors,
                                      'GBps': round(total / log['all_done'] / 1e9, 2), 'events': log}), flush=True)
    return 1 if errors else 0


def cmd_fetch_ckpt(a):
    """The plain HF checkpoint (stock vLLM's input) with the same Run:ai streamer: small files first (then
    <marker-dir>/.config_done), then every *.safetensors in --parts ranged requests, each on its own streamer,
    into <name>.part (mmap'd), all renamed to <name> once every part landed; then <marker-dir>/.weights_done."""
    T0 = time.time()
    log = {}
    stamp = lambda k: log.__setitem__(k, round(time.time() - T0, 3))  # noqa: E731
    if a.src.startswith('gs://'):
        bucket, prefix = split_gs(a.src)
        files = gcs_list(bucket, prefix)
        url = lambda name: f'gs://{bucket}/{prefix}/{name}'  # noqa: E731
        small = lambda name: gcs_bytes(bucket, f'{prefix}/{name}')  # noqa: E731
    else:
        files = [(n, os.path.getsize(os.path.join(a.src, n))) for n in sorted(os.listdir(a.src))
                 if os.path.isfile(os.path.join(a.src, n))]
        url = lambda name: os.path.join(a.src, name)  # noqa: E731
        small = lambda name: open(os.path.join(a.src, name), 'rb').read()  # noqa: E731, SIM115
    stamp('listed')
    os.makedirs(a.dst, exist_ok=True)
    os.makedirs(a.marker_dir, exist_ok=True)
    big = [(n, s) for n, s in files if n.endswith('.safetensors')]
    for n, s in files:
        if not n.endswith('.safetensors'):
            data = small(n)
            if len(data) != s:
                raise RuntimeError(f'{n}: {len(data)} bytes, expected {s}')
            with open(os.path.join(a.dst, n + '.tmp'), 'wb') as fh:
                fh.write(data)
            os.replace(os.path.join(a.dst, n + '.tmp'), os.path.join(a.dst, n))
    open(os.path.join(a.marker_dir, '.config_done'), 'w').close()
    stamp('config_done')
    print(f'fetch-ckpt: {len(files) - len(big)} small files {log["config_done"]:.2f} s', flush=True)

    errors, lock, left, maps, streamers = [], threading.Lock(), {}, {}, []
    jobs = []
    for n, s in big:
        fd = os.open(os.path.join(a.dst, n + '.part'), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, s)
        maps[n] = mmap.mmap(fd, s)
        os.close(fd)
        step = -(-s // a.parts)
        parts = [(off, min(step, s - off)) for off in range(0, s, step)]
        left[n] = len(parts)
        jobs += [(n, off, ln) for off, ln in parts]

    def part(n, off, ln):
        """One ranged request on its own streamer (stream_into retries what ends early). Every streamer stays
        open until all parts are done: f16's three early FINISHED came within 4 ms of the fourth part's close."""
        try:
            st = Streamer(a.lib)
            with lock:
                streamers.append(st)
            stream_into(st, a.lib, url(n), ln, memoryview(maps[n])[off:off + ln], base=off, spare=streamers,
                        tag=n)
            with lock:
                left[n] -= 1
                last = left[n] == 0
            if last:
                stamp(n + '_done')
                print(f'fetch-ckpt: {n} {log[n + "_done"]:.2f} s', flush=True)
        except BaseException as e:  # noqa: BLE001
            errors.append(repr(e))
            print(f'fetch-ckpt: ERROR {e!r}', flush=True)

    threads = [threading.Thread(target=part, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for st in streamers:
        st.close()
        st.release()
    for n, _ in big:
        maps[n].close()
        if not errors:
            os.replace(os.path.join(a.dst, n + '.part'), os.path.join(a.dst, n))
    if not errors:
        open(os.path.join(a.marker_dir, '.weights_done'), 'w').close()
    stamp('all_done')
    total = sum(s for _, s in files)
    print('FETCH_CKPT_JSON=' + json.dumps({'src': a.src, 'bytes': total, 'seconds': log['all_done'], 'errors': errors,
                                           'GBps': round(total / log['all_done'] / 1e9, 2), 'events': log}), flush=True)
    return 1 if errors else 0



def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('pack')
    for k in ('--model-dir', '--bank-dir', '--xla-dir', '--snap-dir', '--out'):
        s.add_argument(k, required=True)
    s = sub.add_parser('upload')
    s.add_argument('--pack', required=True)
    s.add_argument('--dst', required=True)
    s.add_argument('--workers', type=int, default=32)
    s = sub.add_parser('delete')
    s.add_argument('--dst', required=True)
    s = sub.add_parser('fetch')
    for k in ('--src', '--lib', '--model-dir', '--bank-dir', '--xla-dir', '--snap-dir'):
        s.add_argument(k, required=True)
    s.add_argument('--state-only', action='store_true',
                   help='only state.tar (bank + XLA cache): stock vLLM with its compile cache published to GCS')
    s = sub.add_parser('fetch-ckpt')
    for k in ('--src', '--lib', '--dst', '--marker-dir'):
        s.add_argument(k, required=True)
    s.add_argument('--parts', type=int, default=2, help='ranged requests (one streamer each) per safetensors file')
    a = p.parse_args()
    return {'pack': cmd_pack, 'upload': cmd_upload, 'delete': cmd_delete, 'fetch': cmd_fetch,
            'fetch-ckpt': cmd_fetch_ckpt}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
