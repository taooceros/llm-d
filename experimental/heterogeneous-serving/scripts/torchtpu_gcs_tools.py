#!/usr/bin/env python3
"""Host-side helpers for the GCS arms of the vLLM-TorchTPU load probe (stdlib only; run with the image's python3.12).

  auth URI             metadata-server identity + JSON-API listing of gs://bucket/prefix (bytes, files)
  netmon OUT STOPFILE  sample host RX/TX bytes (/proc/net/dev, all non-lo interfaces; the TPU pod uses
                       hostNetwork, so this is the whole host incl. the gcsfuse sidecar) every 0.5 s until
                       STOPFILE exists
  resident DIR         page-cache residency (mincore) of every file in DIR
  evict DIR            posix_fadvise(DONTNEED) on every file in DIR, then report residency
"""
import ctypes
import json
import os
import sys
import time
import urllib.parse
import urllib.request

MD = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/'


def md(path):
    req = urllib.request.Request(MD + path, headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode()


def auth(uri):
    bucket, _, prefix = uri.removeprefix('gs://').partition('/')
    print('sa_email', md('email'), flush=True)
    token = json.loads(md('token'))['access_token']
    items, page = [], None
    while True:
        q = {'prefix': prefix.rstrip('/') + '/', 'fields': 'items(name,size),nextPageToken'}
        if page:
            q['pageToken'] = page
        req = urllib.request.Request(
            f'https://storage.googleapis.com/storage/v1/b/{bucket}/o?' + urllib.parse.urlencode(q),
            headers={'Authorization': f'Bearer {token}'})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        items += d.get('items', [])
        page = d.get('nextPageToken')
        if not page:
            break
    st = [i for i in items if i['name'].endswith('.safetensors')]
    print('GCS_AUTH_JSON=' + json.dumps({
        'uri': uri, 'objects': len(items), 'safetensors_files': len(st),
        'safetensors_bytes': sum(int(i['size']) for i in st),
        'all_bytes': sum(int(i['size']) for i in items)}), flush=True)


def rxtx():
    rx = tx = 0
    with open('/proc/net/dev') as f:
        for line in f.readlines()[2:]:
            name, _, rest = line.partition(':')
            if name.strip() == 'lo':
                continue
            v = rest.split()
            rx += int(v[0])
            tx += int(v[8])
    return rx, tx


def netmon(out, stop):
    with open(out, 'w') as f:
        f.write('epoch\trx_bytes\ttx_bytes\n')
        while not os.path.exists(stop):
            rx, tx = rxtx()
            f.write(f'{time.time():.3f}\t{rx}\t{tx}\n')
            f.flush()
            time.sleep(0.5)
        rx, tx = rxtx()
        f.write(f'{time.time():.3f}\t{rx}\t{tx}\n')


libc = ctypes.CDLL('libc.so.6', use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = os.sysconf('SC_PAGE_SIZE')


def resident_bytes(path):
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED: maps without faulting pages in
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f'mmap {path}')
        try:
            pages = (size + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f'mincore {path}')
            raw = bytes(vec)
            return (pages - raw.count(0)) * PAGE, size
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def files(d):
    """Regular files under d, recursively (a venv is a tree; a checkpoint dir is flat). Symlinks are skipped."""
    out = []
    for root, _, names in os.walk(d):
        out += [p for p in (os.path.join(root, n) for n in names) if os.path.isfile(p) and not os.path.islink(p)]
    return sorted(out)


def resident(d, label='resident'):
    res = tot = 0
    for p in files(d):
        r, s = resident_bytes(p)
        res += r
        tot += s
    print(f'PAGECACHE_JSON=' + json.dumps({'label': label, 'dir': d, 'resident_bytes': res, 'total_bytes': tot}), flush=True)


def evict(d):
    resident(d, 'before_evict')
    for p in files(d):
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    resident(d, 'after_evict')


if __name__ == '__main__':
    cmd, *rest = sys.argv[1:]
    {'auth': auth, 'netmon': netmon, 'resident': resident, 'evict': evict}[cmd](*rest)
