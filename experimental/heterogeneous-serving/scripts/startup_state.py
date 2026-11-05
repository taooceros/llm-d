#!/usr/bin/env python3
"""Ship one instance's startup state onto the hosts of a congruent placement (RCM-supervised).

    flock /tmp/hetero-cluster.lock .venv/bin/python rcm_exec run scripts/startup_state.py -- fetch \\
        --inventory /tmp/hetero-inventory-only.json --method peer --parts snapshot,bank,xla \\
        --source H:large_a --source-scope <seed> --target H:large_b --target-scope <rid> --output <json>

Startup state of a placement, per rank r, under the roots of a run scope (hetero/run_scope.py):

    snapshot  <weight_snapshot_root>/<layout>/<instance>/rank<r>/   manifest.json + its shard files
    bank      <exec_bank_dir>/<identity digest>/rank<r>/             exactly one digest per rank
    xla       <compilation_cache_dir>/                               every process writes its own entries
                                                                     (single-device programs key on its chips)

The snapshot context, the bank identity and (with JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none,
hetero/layout.py) the XLA cache keys name no hosts or paths, so rank r's files serve rank r of any
congruent placement (same tp, chip box and relative host grid: bank_relocate.congruent_pairs).

Actions:
    upload  the source placement's state -> gs://<bucket>/<prefix>/rank<r>/{snapshot,bank,xla}/
    fetch   onto the target placement's hosts, into --target-scope, by --method:
              gcs   google-cloud-storage transfer_manager downloads (process workers) of the uploaded objects
              fuse  parallel copies of the same objects through the /models GCS-fuse mount
              peer  TCP streams from the source placement's hosts, rank r -> rank r
    delete  every object under gs://<bucket>/<prefix>/
    probe   peer transport check without any run state: --probe-bytes per rank pair of synthetic
            files, source -> target hosts, under /dev/shm/omp_peer_probe (removed afterwards)
    fetch-checkpoint   the vanilla alternative: every target host copies the full HF checkpoint
            gs://<bucket>/<model dir>/ (top-level files) into the run scope's checkpoint dir
            (hetero/run_scope.py scope_checkpoint_dir), for a normal-loader boot from local tmpfs.
            --method gcs (default): sliced transfer_manager downloads; fuse: --workers threads of 64 MiB
            pread/pwrite ranges through the /models GCS-fuse mount
    ckpt-io  checkpoint read characterization, no engine: on every target host at once, per --variants
            entry, evict the checkpoint's page cache and read all of it once (ckpt_io_variant), bytes discarded

Fetch refuses a target that already holds any of the requested parts or a delivery marker. It lands
files under their final names, except for the two publication points, and publishes those last:
the snapshot manifest (after every shard's size is checked and its mtime set to the manifest's
mtime_ns, the v2 identity check) and the bank directory (staged, ``relocation.json`` written,
renamed into place, then sha256-verified by ``exec_bank.read_rank_bank``). An engine therefore never
sees a half-fetched snapshot or bank. After that, each rank's delivery marker is written
(<weight_snapshot_root>/.delivered/<layout>/<instance>/rank<r>.json, hetero/state_delivery.py).
A state-gated worker waits for this marker before its first device touch.
``--print-remote-argv`` prints the remote driver's command line instead of submitting it. The
boot-overlap arms hand it to ``hetero.cli deploy --state-delivery``, which runs the fetch while the
new engine boots. ``snapshot_validate.py cleanup --run-scope`` removes everything a fetch writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import struct
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARTS = ("snapshot", "bank", "xla")
METHODS = ("gcs", "fuse", "peer")
FUSE_MOUNT = Path("/models")
FUSE_BUCKET = "hongtaozhang-tpu-model-cache-us-west1"
SNAPSHOT_MANIFEST = "manifest.json"
INCOMING_SUFFIX = ".incoming"
BANK_INCOMING = ".incoming"
RELOCATION = "relocation.json"
# Record header: file index in the receiver's plan, byte count.  A format string, not a struct.Struct:
# Ray pickles this script's globals by value (it runs as __main__) and Struct objects do not pickle.
HEADER = "!IQ"
HEADER_SIZE = struct.calcsize(HEADER)
END = 0xFFFFFFFF
SOCKET_BUFFER = 32 << 20
COPY_BUFFER = 16 << 20


# ------------------------------------------------------------------ layout of a run scope

def parse_parts(text):
    parts = [p for p in str(text).split(",") if p]
    if not parts or len(set(parts)) != len(parts) or set(parts) - set(PARTS):
        raise ValueError(f"parts: a comma list drawn from {PARTS}, got {text!r}")
    return [p for p in PARTS if p in parts]


def split_gcs(url):
    if not url.startswith("gs://"):
        raise ValueError(f"not a gs:// URL: {url}")
    bucket, _, prefix = url[len("gs://"):].partition("/")
    prefix = prefix.strip("/")
    if not bucket or not prefix:
        raise ValueError(f"gs://<bucket>/<prefix> needs both: {url}")
    return bucket, prefix


def roots(scope):
    from hetero.run_scope import scope_roots, scope_xla_dir
    r = scope_roots(scope)
    return {"snapshot": Path(r["weight_snapshot_root"]), "bank": Path(r["exec_bank_dir"]),
            "xla": Path(scope_xla_dir(scope))}


def snapshot_dir(scope, spec, rank):
    layout, _, instance = spec.partition(":")
    if not layout or not instance:
        raise ValueError(f"LAYOUT:INSTANCE expected, got {spec!r}")
    return roots(scope)["snapshot"] / layout / instance / f"rank{int(rank)}"


def _safe_rel(rel):
    path = Path(rel)
    if not rel or path.is_absolute() or ".." in path.parts or path.parts[0].startswith(BANK_INCOMING):
        raise ValueError(f"unsafe relative name: {rel!r}")
    return path


def source_files(scope, spec, rank, parts):
    """[{part, rel, path, bytes}] of one source rank on this host; ``rel`` is relative to the part."""
    rows = []

    def add(part, rel, path):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{part}: not a regular file: {path}")
        rows.append({"part": part, "rel": str(rel), "path": str(path), "bytes": path.stat().st_size})

    if "snapshot" in parts:
        directory = snapshot_dir(scope, spec, rank)
        manifest = json.loads((directory / SNAPSHOT_MANIFEST).read_text())
        for array in manifest["arrays"]:
            for shard in array["shards"]:
                if Path(shard["file"]).name != shard["file"]:
                    raise RuntimeError(f"snapshot shard outside its directory: {shard['file']}")
                add("snapshot", shard["file"], directory / shard["file"])
        add("snapshot", SNAPSHOT_MANIFEST, directory / SNAPSHOT_MANIFEST)
    if "bank" in parts:
        bank_root = roots(scope)["bank"]
        found = sorted(p.parent for p in bank_root.glob(f"*/rank{int(rank)}/{SNAPSHOT_MANIFEST}")
                       if not p.parent.parent.name.startswith("."))
        if len(found) != 1:
            raise RuntimeError(f"rank {rank}: expected exactly one bank under {bank_root}, found {[str(f) for f in found]}")
        digest = found[0].parent.name
        for path in sorted(found[0].iterdir()):
            if path.name != RELOCATION:  # a relocated source's own provenance is not part of the bank
                add("bank", f"{digest}/{path.name}", path)
    if "xla" in parts:
        base = roots(scope)["xla"]
        if not base.is_dir():
            raise RuntimeError(f"no XLA cache directory {base}")
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.is_symlink():
                add("xla", path.relative_to(base), path)
        if not any(r["part"] == "xla" for r in rows):
            raise RuntimeError(f"empty XLA cache directory {base}")
    return rows


def dest_path(scope, spec, rank, part, rel):
    """Where a fetch lands ``rel`` of ``part`` for target rank ``rank`` (publication points staged)."""
    rel_path = _safe_rel(rel)
    r = roots(scope)
    if part == "snapshot":
        if len(rel_path.parts) != 1:
            raise ValueError(f"snapshot files are flat: {rel!r}")
        name = rel + INCOMING_SUFFIX if rel == SNAPSHOT_MANIFEST else rel
        return snapshot_dir(scope, spec, rank) / name
    if part == "bank":
        if len(rel_path.parts) != 2:
            raise ValueError(f"bank names are <digest>/<file>: {rel!r}")
        digest, name = rel_path.parts
        return r["bank"] / BANK_INCOMING / digest / f"rank{int(rank)}" / name
    if part == "xla":
        return r["xla"] / rel_path
    raise ValueError(f"unknown part {part!r}")


def gcs_object(prefix, rank, part, rel):
    _safe_rel(rel)
    return f"{prefix}/rank{int(rank)}/{part}/{rel}"


def gcs_part_prefix(prefix, rank, part):
    return f"{prefix}/rank{int(rank)}/{part}/"


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tree(path):
    files = [p for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink()] if Path(path).is_dir() else []
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def delivery_marker(scope, spec, rank):
    """The rank's "startup state published" marker a state-gated worker waits for (hetero/state_delivery.py)."""
    from hetero.state_delivery import marker_path
    layout, _, instance = spec.partition(":")
    return marker_path(str(roots(scope)["snapshot"]), layout, instance, rank)


def prepare_target(scope, spec, rank, parts):
    """Refuse a target rank that already holds any requested part or a delivery marker; report what was there."""
    r = roots(scope)
    state = {}
    marker = delivery_marker(scope, spec, rank)
    if marker.exists():
        raise RuntimeError(f"target already holds a delivery marker: {marker}")
    if "snapshot" in parts:
        directory = snapshot_dir(scope, spec, rank)
        state["snapshot"] = _tree(directory)
        if state["snapshot"]["files"]:
            raise RuntimeError(f"target snapshot directory is not empty: {directory}")
    if "bank" in parts:
        present = sorted(str(p) for p in r["bank"].glob(f"*/rank{int(rank)}"))
        state["bank"] = {"rank_banks": present}
        if present:
            raise RuntimeError(f"target already holds a bank for rank {rank}: {present}")
    if "xla" in parts:
        state["xla"] = _tree(r["xla"])
        if state["xla"]["files"]:
            raise RuntimeError(f"target XLA cache directory is not empty: {r['xla']}")
    return {"host": socket.gethostname(), "rank": int(rank), "before": state}


def finalize(scope, spec, rank, parts, provenance):
    """Publish what a fetch landed for one target rank on this host (see the module docstring)."""
    from hetero import exec_bank
    started = time.time()
    report = {"rank": int(rank), "host": socket.gethostname()}
    if "snapshot" in parts:
        directory = snapshot_dir(scope, spec, rank)
        incoming = directory / (SNAPSHOT_MANIFEST + INCOMING_SUFFIX)
        manifest = json.loads(incoming.read_text())
        shards = total = 0
        for array in manifest["arrays"]:
            for shard in array["shards"]:
                path = directory / shard["file"]
                if path.stat().st_size != shard["bytes"]:
                    raise RuntimeError(f"fetched shard has the wrong size: {path}")
                if "mtime_ns" in shard:
                    os.utime(path, ns=(int(shard["mtime_ns"]), int(shard["mtime_ns"])))
                shards += 1
                total += shard["bytes"]
        os.replace(incoming, directory / SNAPSHOT_MANIFEST)
        _sync_directory(directory)
        report["snapshot"] = {"directory": str(directory), "shards": shards, "bytes": total,
                              "generation": manifest.get("generation"), "context": manifest.get("context")}
    if "bank" in parts:
        bank_root = roots(scope)["bank"]
        staged = sorted((bank_root / BANK_INCOMING).glob(f"*/rank{int(rank)}"))
        if len(staged) != 1:
            raise RuntimeError(f"expected one staged bank for rank {rank}, found {[str(s) for s in staged]}")
        digest = staged[0].parent.name
        final = bank_root / digest / f"rank{int(rank)}"
        if final.exists():
            raise RuntimeError(f"refusing to replace an existing bank: {final}")
        raw = (staged[0] / SNAPSHOT_MANIFEST).read_bytes()
        (staged[0] / RELOCATION).write_text(json.dumps(
            {**provenance, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "installed_at": time.time()},
            sort_keys=True, indent=1))
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staged[0], final)
        _sync_directory(final.parent)
        for leftover in (staged[0].parent, bank_root / BANK_INCOMING):
            try:
                leftover.rmdir()
            except OSError:
                pass
        verified = exec_bank.read_rank_bank(str(bank_root), digest, int(rank))  # sha256 of every entry
        report["bank"] = {"path": str(final), "identity_digest": digest, "entries": len(verified["files"]),
                          **_tree(final)}
    if "xla" in parts:
        report["xla"] = {"directory": str(roots(scope)["xla"]), **_tree(roots(scope)["xla"])}
    # Last: a state-gated worker on this host starts its device init as soon as this appears.
    from hetero.state_delivery import write_marker
    marker = delivery_marker(scope, spec, rank)
    write_marker(marker, {"rank": int(rank), "host": report["host"], "parts": list(parts),
                          "provenance": provenance, "published_at": time.time()})
    report["marker"] = str(marker)
    report["finalize_s"] = time.time() - started
    return report


# ------------------------------------------------------------------ peer transport

def _recv_exact(conn, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = conn.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the stream inside a header")
        got += k
    return bytes(buf)


class Receiver:
    """Target-host end of a peer fetch: ``connections`` TCP streams of (index, size, bytes) records,
    each written to ``plan[index]`` (a [path, bytes] pair); every plan entry must arrive exactly once."""

    def __init__(self, plan, connections, timeout):
        self.plan = [(str(p), int(b)) for p, b in plan]
        self.connections = int(connections)
        self.timeout = float(timeout)
        self.server = socket.create_server(("", 0), backlog=max(self.connections, 1))
        self.server.settimeout(self.timeout)

    def port(self):
        return self.server.getsockname()[1]

    def run(self):
        started = time.time()
        seen = [False] * len(self.plan)
        lock = threading.Lock()
        errors, stats = [], {"bytes": 0, "files": 0}

        def handle(conn):
            try:
                conn.settimeout(self.timeout)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER)
                buf = bytearray(COPY_BUFFER)
                view = memoryview(buf)
                while True:
                    index, size = struct.unpack(HEADER, _recv_exact(conn, HEADER_SIZE))
                    if index == END:
                        return
                    if index >= len(self.plan) or self.plan[index][1] != size:
                        raise RuntimeError(f"unexpected record {index} of {size} bytes")
                    with lock:
                        if seen[index]:
                            raise RuntimeError(f"record {index} arrived twice")
                        seen[index] = True
                    path = Path(self.plan[index][0])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                    try:
                        left = size
                        while left:
                            k = conn.recv_into(view, min(len(buf), left))
                            if k == 0:
                                raise ConnectionError(f"peer closed the stream inside {path}")
                            off = 0
                            while off < k:
                                off += os.write(fd, view[off:k])
                            left -= k
                    finally:
                        os.close(fd)
                    with lock:
                        stats["bytes"] += size
                        stats["files"] += 1
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                conn.close()

        threads = []
        try:
            for _ in range(self.connections):
                conn, _ = self.server.accept()
                thread = threading.Thread(target=handle, args=(conn,), daemon=True)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join(self.timeout)
        finally:
            self.server.close()
        missing = [self.plan[i][0] for i, ok in enumerate(seen) if not ok]
        if errors or missing or any(t.is_alive() for t in threads):
            raise RuntimeError(f"peer receive failed: errors={errors[:5]} missing={len(missing)} {missing[:3]}")
        return {**stats, "seconds": time.time() - started, "connections": self.connections,
                "host": socket.gethostname()}


def send_files(host, port, items, streams, timeout):
    """Source-host end: ``items`` [[index, path, bytes]] over ``streams`` byte-balanced TCP streams."""
    lanes, loads = [[] for _ in range(streams)], [0] * streams
    for item in sorted(items, key=lambda i: -int(i[2])):
        k = loads.index(min(loads))
        lanes[k].append(item)
        loads[k] += int(item[2])
    errors = []
    started = time.time()

    def lane(rows):
        try:
            with socket.create_connection((host, int(port)), timeout=timeout) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER)
                for index, path, size in rows:
                    with open(path, "rb") as f:
                        if os.fstat(f.fileno()).st_size != int(size):
                            raise RuntimeError(f"source file changed size: {path}")
                        sock.sendall(struct.pack(HEADER, int(index), int(size)))
                        if int(size):
                            sent = sock.sendfile(f, 0, int(size))
                            if sent != int(size):
                                raise RuntimeError(f"short send of {path}: {sent}/{size}")
                sock.sendall(struct.pack(HEADER, END, 0))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=lane, args=(rows,), daemon=True) for rows in lanes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)
    if errors or any(t.is_alive() for t in threads):
        raise RuntimeError(f"peer send to {host}:{port} failed: {errors[:5]}")
    return {"bytes": sum(loads), "files": len(items), "streams": streams, "seconds": time.time() - started,
            "host": socket.gethostname()}


# ------------------------------------------------------------------ GCS transport

def _storage():
    from google.cloud import storage
    return storage.Client()


def gcs_upload(bucket, prefix, scope, spec, rank, parts, workers, worker_type="process"):
    from google.cloud.storage import transfer_manager as tm
    files = source_files(scope, spec, rank, parts)
    client = _storage()
    b = client.bucket(bucket)
    started = time.time()
    results = tm.upload_many([(f["path"], b.blob(gcs_object(prefix, rank, f["part"], f["rel"]))) for f in files],
                             max_workers=workers, worker_type=worker_type, raise_exception=True,
                             upload_kwargs={"checksum": "crc32c"})
    failed = [repr(r) for r in results if isinstance(r, Exception)]
    if failed:
        raise RuntimeError(f"upload failures: {failed[:3]}")
    return {"rank": int(rank), "host": socket.gethostname(), "files": len(files),
            "bytes": sum(f["bytes"] for f in files), "seconds": time.time() - started,
            "by_part": {p: sum(f["bytes"] for f in files if f["part"] == p) for p in parts}}


def _gcs_listing(bucket, prefix, rank, parts):
    client = _storage()
    rows = []
    for part in parts:
        head = gcs_part_prefix(prefix, rank, part)
        for blob in client.list_blobs(bucket, prefix=head):
            rows.append((part, blob.name[len(head):], blob))
    return rows


def gcs_fetch(bucket, prefix, scope, spec, rank, parts, workers, worker_type="process"):
    from google.cloud.storage import transfer_manager as tm
    listed = time.time()
    rows = _gcs_listing(bucket, prefix, rank, parts)
    _require_parts(rows, parts, f"gs://{bucket}/{prefix} rank {rank}")
    pairs = []
    for part, rel, blob in rows:
        path = dest_path(scope, spec, rank, part, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        pairs.append((blob, str(path)))
    started = time.time()
    results = tm.download_many(pairs, max_workers=workers, worker_type=worker_type, raise_exception=True,
                               download_kwargs={"checksum": "crc32c"})
    failed = [repr(r) for r in results if isinstance(r, Exception)]
    if failed:
        raise RuntimeError(f"download failures: {failed[:3]}")
    return {"rank": int(rank), "host": socket.gethostname(), "files": len(rows),
            "bytes": sum(blob.size for _, _, blob in rows), "list_s": started - listed,
            "seconds": time.time() - started,
            "by_part": {p: sum(b.size for q, _, b in rows if q == p) for p in parts}}


def fuse_fetch(bucket, prefix, scope, spec, rank, parts, workers):
    from concurrent.futures import ThreadPoolExecutor
    if bucket != FUSE_BUCKET:
        raise RuntimeError(f"only gs://{FUSE_BUCKET} is mounted at {FUSE_MOUNT}")
    listed = time.time()
    rows = []
    for part in parts:
        head = FUSE_MOUNT / gcs_part_prefix(prefix, rank, part)
        for path in sorted(head.rglob("*")):
            if path.is_file():
                rows.append((part, str(path.relative_to(head)), path))
    _require_parts(rows, parts, f"{FUSE_MOUNT}/{prefix} rank {rank}")
    jobs = []
    for part, rel, path in rows:
        dest = dest_path(scope, spec, rank, part, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((path, dest))
    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        sizes = list(pool.map(lambda job: (shutil.copyfile(*job), job[1].stat().st_size)[1], jobs))
    return {"rank": int(rank), "host": socket.gethostname(), "files": len(rows), "bytes": sum(sizes),
            "list_s": started - listed, "seconds": time.time() - started,
            "by_part": {p: sum(s for (q, _, _), s in zip(rows, sizes) if q == p) for p in parts}}


def _require_parts(rows, parts, where):
    missing = [p for p in parts if not any(r[0] == p for r in rows)]
    if missing:
        raise RuntimeError(f"{where}: no objects for {missing}")
    if "snapshot" in parts and not any(r[0] == "snapshot" and r[1] == SNAPSHOT_MANIFEST for r in rows):
        raise RuntimeError(f"{where}: snapshot without its manifest")


def gcs_delete(bucket, prefix):
    client = _storage()
    blobs = list(client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"))
    total = sum(b.size or 0 for b in blobs)
    for start in range(0, len(blobs), 100):
        with client.batch():
            for blob in blobs[start:start + 100]:
                blob.delete()
    return {"objects": len(blobs), "bytes": total, "host": socket.gethostname()}


def gcs_absent(bucket, prefix):
    return not any(True for _ in _storage().list_blobs(bucket, prefix=prefix.rstrip("/") + "/", max_results=1))


def checkpoint_fetch(bucket, prefix, scope, workers, worker_type="process", chunk_bytes=64 << 20):
    """Download every top-level object of gs://bucket/prefix/ into <scope checkpoint dir>/<basename>/."""
    from hetero.run_scope import scope_checkpoint_dir
    return gcs_download_dir(bucket, prefix, Path(scope_checkpoint_dir(scope)) / Path(prefix).name, workers,
                            worker_type, chunk_bytes)


def gcs_download_dir(bucket, prefix, dest, workers, worker_type="process", chunk_bytes=64 << 20):
    """Sliced transfer_manager download of every top-level object of gs://bucket/prefix/ into dest/."""
    from google.cloud.storage import transfer_manager as tm
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"target checkpoint directory is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    client = _storage()
    listed = time.time()
    blobs = [b for b in client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/")
             if "/" not in b.name[len(prefix.rstrip("/")) + 1:] and b.size]
    if not any(b.name.endswith(".safetensors") for b in blobs):
        raise RuntimeError(f"gs://{bucket}/{prefix}: no safetensors objects")
    started = time.time()
    small = [b for b in blobs if b.size < 4 * chunk_bytes]
    if small:
        failed = [r for r in tm.download_many([(b, str(dest / Path(b.name).name)) for b in small],
                                              max_workers=workers, worker_type=worker_type, raise_exception=True)
                  if isinstance(r, Exception)]
        if failed:
            raise RuntimeError(f"download failures: {failed[:3]!r}")
    files = []
    for blob in (b for b in blobs if b.size >= 4 * chunk_bytes):
        began = time.time()
        tm.download_chunks_concurrently(blob, str(dest / Path(blob.name).name), chunk_size=chunk_bytes,
                                        max_workers=workers, worker_type=worker_type)
        files.append({"name": Path(blob.name).name, "bytes": blob.size, "seconds": time.time() - began})
    for blob in blobs:
        if (dest / Path(blob.name).name).stat().st_size != blob.size:
            raise RuntimeError(f"downloaded size mismatch: {blob.name}")
    total = sum(b.size for b in blobs)
    return {"host": socket.gethostname(), "directory": str(dest), "files": len(blobs), "bytes": total,
            "list_s": started - listed, "seconds": time.time() - started, "large_files": files}


# ------------------------------------------------------------------ checkpoint reads through GCS FUSE

CKPT_CHUNK = 64 << 20


def fuse_checkpoint_files(prefix, suffixes=None):
    """[(path, bytes)] of the non-empty top-level files of the checkpoint as the /models mount shows it."""
    source = FUSE_MOUNT / prefix.strip("/")
    files = [(str(p), p.stat().st_size) for p in sorted(source.iterdir())
             if p.is_file() and p.stat().st_size and (suffixes is None or p.suffix in suffixes)]
    if not any(p.endswith(".safetensors") for p, _ in files):
        raise RuntimeError(f"{source}: no safetensors files")
    return files


def chunk_jobs(files, chunk):
    """(path, bytes) -> [(path, offset, length)] covering every byte once, file by file."""
    return [(path, offset, min(chunk, size - offset)) for path, size in files for offset in range(0, size, chunk)]


def read_chunks(jobs, workers, sink=None, shared_fd=False):
    """pread every (path, offset, length) on ``workers`` threads; ``sink(path, offset, view)`` consumes it.

    Each read opens its own descriptor unless ``shared_fd``: then all threads pread one descriptor per
    file (as fastsafetensors' reader does)."""
    from concurrent.futures import ThreadPoolExecutor
    local = threading.local()
    shared = {path: os.open(path, os.O_RDONLY) for path in {job[0] for job in jobs}} if shared_fd else {}

    def one(job):
        path, offset, length = job
        if getattr(local, "buf", None) is None or len(local.buf) < length:
            local.buf = bytearray(max(length, CKPT_CHUNK))
        view = memoryview(local.buf)[:length]
        fd = shared[path] if shared_fd else os.open(path, os.O_RDONLY)
        try:
            got = 0
            while got < length:
                n = os.preadv(fd, [view[got:]], offset + got)
                if n <= 0:
                    raise RuntimeError(f"short read: {path} at {offset + got}")
                got += n
        finally:
            if not shared_fd:
                os.close(fd)
        if sink is not None:
            sink(path, offset, view)
        return length

    try:
        with ThreadPoolExecutor(max(1, int(workers))) as pool:
            return sum(pool.map(one, jobs))
    finally:
        for fd in shared.values():
            os.close(fd)


def checkpoint_fuse_fetch(prefix, scope, workers, chunk_bytes=CKPT_CHUNK):
    """Copy the checkpoint from the /models GCS FUSE mount into <scope checkpoint dir>/<basename>/:
    ``workers`` threads, each pread/pwrite-ing 64 MiB ranges (large files stream in parallel too)."""
    from hetero.run_scope import scope_checkpoint_dir
    dest = Path(scope_checkpoint_dir(scope)) / Path(prefix).name
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f"target checkpoint directory is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    listed = time.time()
    files = fuse_checkpoint_files(prefix)
    started = time.time()
    fds = {}
    try:
        for path, size in files:
            fd = os.open(dest / Path(path).name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            fds[path] = fd
            os.posix_fallocate(fd, 0, size)  # tmpfs: ENOSPC now, not half-way through

        def sink(path, offset, view):
            done = 0
            while done < len(view):
                done += os.pwrite(fds[path], view[done:], offset + done)

        total = read_chunks(chunk_jobs(files, chunk_bytes), workers, sink)
    finally:
        for fd in fds.values():
            os.close(fd)
    for path, size in files:
        if (dest / Path(path).name).stat().st_size != size:
            raise RuntimeError(f"copied size mismatch: {path}")
    return {"host": socket.gethostname(), "directory": str(dest), "files": len(files), "bytes": total,
            "list_s": started - listed, "seconds": time.time() - started, "workers": int(workers)}


IO_ROOT = "/dev/shm/omp_ckpt_io"
IO_VARIANTS = ("st-lazy", "seq", "par-8", "par-32", "par-64", "gcs-32")


def safetensors_touch(paths):
    """The default loader's checkpoint access without its conversion: files in sorted order, one thread,
    safe_open(framework='pt') and get_tensor per key in key order (vllm safetensors_weights_iterator);
    clone() faults in every byte of each tensor the way the conversion's first read does."""
    from safetensors import safe_open
    total = 0
    for path in sorted(paths):
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name).clone()
                total += tensor.numel() * tensor.element_size()
                del tensor
    return total


def evict_files(paths):
    """posix_fadvise(DONTNEED) each file: its next read starts cold (as snapshot_validate evict-pagecache)."""
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def resident_fraction(paths):
    """Fraction of the files' bytes in the page cache (mincore, as snapshot_validate checkpoint_residency:
    a private, never-written map faults nothing, so this reads no file bytes)."""
    import ctypes
    import mmap
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    page, total, resident = os.sysconf("SC_PAGE_SIZE"), 0, 0
    for path in paths:
        size = os.path.getsize(path)
        if not size:
            continue
        vector = (ctypes.c_ubyte * ((size + page - 1) // page))()
        with open(path, "rb") as handle, mmap.mmap(handle.fileno(), size, flags=mmap.MAP_PRIVATE,
                                                     prot=mmap.PROT_READ | mmap.PROT_WRITE) as mapped:
            anchor = ctypes.c_char.from_buffer(mapped)
            try:
                if libc.mincore(ctypes.addressof(anchor), size, vector) != 0:
                    raise OSError(ctypes.get_errno(), f"mincore failed for {path}")
            finally:
                del anchor
        total += size
        resident += min(size, sum(b & 1 for b in bytes(vector)) * page)
    return resident / total if total else 0.0


def ckpt_io_variant(bucket, prefix, variant, worker_type="process"):
    """One cold read of the whole checkpoint on this host by ``variant``; bytes are discarded.

    st-lazy   safetensors_touch over the FUSE mount (the default loader's I/O, no conversion)
    seq       one thread, 64 MiB sequential preads, file by file, over the FUSE mount
    par-N     N threads of 64 MiB range preads over the FUSE mount (all files' ranges in one pool)
    gcs-N     transfer_manager sliced download with N process workers into IO_ROOT (tmpfs), then
              safetensors_touch on that local copy (``local_touch_s``: the access pattern without network)
    vpf       vLLM's safetensors prefetch alone (vllm weight_utils._prefetch_all_checkpoints at its defaults:
              8-thread pool, one task per file, 16 MiB sequential reads; ``files``: per-file seconds)
    st-lazy+B safetensors_touch with background reader B (vpf or par-N) started at the same instant;
              ``seconds`` is the walk, ``background_s`` when B finished (the loader waits on the walk only)
    fst[-N]   fastsafetensors (vLLM load_format=fastsafetensors) on the host CPU: SafeTensorsFileLoader
              with nogds pread + bounce buffer, N copy threads (default 16, its default), then
              get_tensor of every key; the library is pip-installed into FST_ROOT outside the timing.
              Its O_DIRECT policy treats only listed network file systems (not fuse) as remote, so on
              GCS FUSE it reads with O_DIRECT, bypassing kernel readahead
    fstb[-N]  the same with FASTSAFETENSORS_ODIRECT=0 (buffered reads)
    sfd-N     N threads of 64 MiB preads sharing one descriptor per file, first SFD_CHUNKS ranges only
    ofd-N     the same ranges with a descriptor per read (as par-N)
    (vLLM's other alternative loader, instanttensor, refuses non-CUDA devices, so it has no TPU variant)
    """
    files = fuse_checkpoint_files(prefix, suffixes=(".safetensors",))
    paths = [p for p, _ in files]
    evict_files(paths)
    row = {"variant": variant, "host": socket.gethostname(), "resident_before": resident_fraction(paths)}
    started = time.time()
    if variant == "st-lazy":
        row["bytes"] = safetensors_touch(paths)
    elif variant == "vpf":
        row["files"] = vllm_prefetch(paths)
        row["bytes"] = sum(f["bytes"] for f in row["files"])
    elif variant.startswith("st-lazy+"):
        background = variant[len("st-lazy+"):]
        result = {}

        def run_background():
            began = time.time()
            if background == "vpf":
                result["files"] = vllm_prefetch(paths)
            else:
                read_chunks(chunk_jobs(files, CKPT_CHUNK), int(background[4:]))
            result["seconds"] = time.time() - began

        thread = threading.Thread(target=run_background, daemon=True)
        thread.start()
        row["bytes"] = safetensors_touch(paths)
        row["seconds"] = time.time() - started
        thread.join()
        row["background_s"] = result.get("seconds")
        row["background_files"] = result.get("files")
    elif variant == "seq":
        row["bytes"] = read_chunks(chunk_jobs(files, CKPT_CHUNK), 1)
    elif variant.startswith("par-"):
        row["bytes"] = read_chunks(chunk_jobs(files, CKPT_CHUNK), int(variant[4:]))
    elif variant.split("-")[0] in ("sfd", "ofd"):
        # fd-sharing control: the first SFD_CHUNKS 64 MiB ranges of the checkpoint on N threads, every
        # thread on one shared descriptor per file (sfd, as fastsafetensors) or its own per read (ofd)
        kind, _, count = variant.partition("-")
        jobs = chunk_jobs(files, CKPT_CHUNK)[:SFD_CHUNKS]
        row["bytes"] = read_chunks(jobs, int(count), shared_fd=kind == "sfd")
    elif variant.startswith("gcs-"):
        scratch = Path(IO_ROOT) / Path(prefix).name
        shutil.rmtree(IO_ROOT, ignore_errors=True)
        try:
            got = gcs_download_dir(bucket, prefix, scratch, int(variant[4:]), worker_type)
            row["seconds"] = got["seconds"]
            local = sorted(str(p) for p in scratch.glob("*.safetensors"))
            touched = time.time()
            row["local_touch_bytes"] = safetensors_touch(local)
            row["local_touch_s"] = time.time() - touched
            row["bytes"] = sum(os.path.getsize(p) for p in local)
        finally:
            shutil.rmtree(IO_ROOT, ignore_errors=True)
    elif variant.split("-")[0] in ("fst", "fstb"):
        kind, _, count = variant.partition("-")
        row.update(fastsafetensors_read(paths, int(count) if count else FST_THREADS, buffered=kind == "fstb"))
    else:
        raise ValueError(f"unknown checkpoint I/O variant {variant!r}")
    row.setdefault("seconds", time.time() - started)
    row["gbps"] = row["bytes"] / row["seconds"] / 1e9
    evict_files(paths)
    return row


VPF_THREADS, VPF_BLOCK = 8, 16 << 20  # vllm.config.load DEFAULT_SAFETENSORS_PREFETCH_{NUM_THREADS,BLOCK_SIZE}


def vllm_prefetch(paths):
    """vLLM's prefetch: a VPF_THREADS pool, one task per file, ``while f.read(VPF_BLOCK)`` (buffered Python
    reads, as vllm weight_utils._prefetch_checkpoint).  Returns [{name, bytes, seconds}] in file order."""
    from concurrent.futures import ThreadPoolExecutor
    began = time.time()

    def one(path):
        total = 0
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(VPF_BLOCK)
                if not chunk:
                    break
                total += len(chunk)
        return {"name": Path(path).name, "bytes": total, "seconds": time.time() - began}

    with ThreadPoolExecutor(VPF_THREADS) as pool:
        return list(pool.map(one, sorted(paths)))


FST_VERSION, FST_THREADS, FST_ROOT = "0.5.0", 16, "/dev/shm/omp_fst"
SFD_CHUNKS = 64  # 4 GiB: enough for a steady rate at ~5 MB/s without an hour per host


def fastsafetensors_read(paths, threads, buffered=False):
    """Read ``paths`` into host memory with fastsafetensors (CPU device, no GDS) and fetch every tensor."""
    import subprocess
    if buffered:
        os.environ["FASTSAFETENSORS_ODIRECT"] = "0"
    else:
        os.environ.pop("FASTSAFETENSORS_ODIRECT", None)
    if not Path(FST_ROOT, "fastsafetensors").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", FST_ROOT,
                        f"fastsafetensors=={FST_VERSION}", "typer"], check=True)
    if FST_ROOT not in sys.path:
        sys.path.insert(0, FST_ROOT)
    from fastsafetensors import SafeTensorsFileLoader
    began = time.time()
    loader = SafeTensorsFileLoader(None, "cpu", nogds=True, max_threads=threads)
    try:
        loader.add_filenames({0: sorted(paths)})
        buffers = loader.copy_files_to_device()
        copied = time.time() - began
        total = 0
        for key in loader.get_keys():
            tensor = buffers.get_tensor(key)
            total += tensor.numel() * tensor.element_size()
        buffers.close()
    finally:
        loader.close()
    return {"bytes": sum(os.path.getsize(p) for p in paths), "tensor_bytes": total,
            "copy_s": copied, "seconds": time.time() - began, "threads": threads, "buffered": buffered}


def io_cleanup():
    shutil.rmtree(IO_ROOT, ignore_errors=True)
    shutil.rmtree(FST_ROOT, ignore_errors=True)
    return {"host": socket.gethostname(), "absent_after": not Path(IO_ROOT).exists() and not Path(FST_ROOT).exists()}


# ------------------------------------------------------------------ remote driver (a Ray job)

def remote(args):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from hetero.evidence import emit_measurements
    ray.init(address="auto", ignore_reinit_error=True)
    nodes = {n["NodeManagerAddress"]: n["NodeID"] for n in ray.nodes() if n["Alive"]}
    tpu_hosts = sorted(n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("TPU"))

    def on(ip):
        if ip not in nodes:
            raise RuntimeError(f"host {ip} is not a live Ray node")
        return NodeAffinitySchedulingStrategy(nodes[ip], soft=False)

    def task(fn):
        return ray.remote(num_cpus=0, max_retries=0)(fn)

    pairs = json.loads(args.pairs) if args.pairs else []
    parts = parse_parts(args.parts)
    bucket, prefix = split_gcs(args.gcs) if args.gcs else (None, None)
    started = time.time()
    common = {"ok": True, "action": args.action, "parts": parts, "source": args.source, "target": args.target,
              "source_scope": args.source_scope, "target_scope": args.target_scope, "gcs": args.gcs}

    if args.action == "delete":
        report = ray.get(task(gcs_delete).options(scheduling_strategy=on(tpu_hosts[0])).remote(bucket, prefix),
                         timeout=args.timeout)
        emit_measurements({**common, **report, "wall_s": time.time() - started})
        return

    if args.action == "fetch-checkpoint":
        method = args.method or "gcs"
        if method == "fuse":
            if bucket != FUSE_BUCKET:
                raise RuntimeError(f"only gs://{FUSE_BUCKET} is mounted at {FUSE_MOUNT}")
            refs = [task(checkpoint_fuse_fetch).options(scheduling_strategy=on(p["target_host"])).remote(
                prefix, args.target_scope, args.workers) for p in pairs]
        else:
            refs = [task(checkpoint_fetch).options(scheduling_strategy=on(p["target_host"])).remote(
                bucket, prefix, args.target_scope, args.workers, args.worker_type) for p in pairs]
        rows = ray.get(refs, timeout=args.timeout)
        for row, pair in zip(rows, pairs):
            row.update(rank=pair["rank"], target_host=pair["target_host"])
        wall = time.time() - started
        total = sum(r["bytes"] for r in rows)
        emit_measurements({**common, "method": f"{method}-checkpoint", "workers": args.workers, "ranks": rows,
                           "bytes": total, "files": sum(r["files"] for r in rows), "wall_s": wall,
                           "transfer_s": wall, "finalize_s": 0.0, "transfer_gbps": total / wall / 1e9,
                           "per_host_gbps": [r["bytes"] / r["seconds"] / 1e9 for r in rows],
                           "directory": rows[0]["directory"] if rows else None})
        return

    if args.action == "ckpt-io":
        if bucket != FUSE_BUCKET:
            raise RuntimeError(f"only gs://{FUSE_BUCKET} is mounted at {FUSE_MOUNT}")
        variants = {}
        try:
            for variant in args.variants.split(","):
                began = time.time()
                rows = ray.get([task(ckpt_io_variant).options(scheduling_strategy=on(p["target_host"])).remote(
                    bucket, prefix, variant, args.worker_type) for p in pairs], timeout=args.timeout)
                for row, pair in zip(rows, pairs):
                    row.update(rank=pair["rank"], target_host=pair["target_host"])
                variants[variant] = {"ranks": rows, "wall_s": time.time() - began,
                                     "per_host_gbps": [r["gbps"] for r in rows],
                                     "max_host_s": max(r["seconds"] for r in rows)}
                print(json.dumps({variant: {k: v for k, v in variants[variant].items() if k != "ranks"}}), flush=True)
        finally:
            ray.get([task(io_cleanup).options(scheduling_strategy=on(p["target_host"])).remote() for p in pairs],
                    timeout=300)
        emit_measurements({**common, "variants": variants, "wall_s": time.time() - started})
        return

    if args.action == "probe":
        chunk = 256 << 20
        sizes = [chunk] * (args.probe_bytes // chunk) + ([args.probe_bytes % chunk] if args.probe_bytes % chunk else [])
        try:
            sources = ray.get([task(probe_write).options(scheduling_strategy=on(p["source_host"])).remote(
                p["rank"], sizes) for p in pairs], timeout=600)
            receiver_cls = ray.remote(num_cpus=0, max_restarts=0)(Receiver)
            receivers = [receiver_cls.options(scheduling_strategy=on(p["target_host"])).remote(
                [[str(Path(PROBE_ROOT) / f"dst-rank{p['rank']}" / f"f{i}"), s] for i, s in enumerate(sizes)],
                args.streams, args.timeout) for p in pairs]
            ports = ray.get([r.port.remote() for r in receivers], timeout=120)
            transfer_started = time.time()
            runs = [r.run.remote() for r in receivers]
            sent = ray.get([task(send_files).options(scheduling_strategy=on(p["source_host"])).remote(
                p["target_host"], port, [[i, path, s] for i, (path, s) in enumerate(zip(files, sizes))],
                args.streams, args.timeout) for p, port, files in zip(pairs, ports, sources)], timeout=args.timeout)
            received = ray.get(runs, timeout=args.timeout)
            transfer_s = time.time() - transfer_started
        finally:
            hosts = sorted({h for p in pairs for h in (p["source_host"], p["target_host"])})
            ray.get([task(probe_cleanup).options(scheduling_strategy=on(h)).remote() for h in hosts], timeout=300)
        total = sum(r["bytes"] for r in received)
        emit_measurements({**common, "pairs": pairs, "senders": sent, "receivers": received, "bytes": total,
                           "streams": args.streams, "transfer_s": transfer_s, "transfer_gbps": total / transfer_s / 1e9,
                           "per_pair_gbps": [r["bytes"] / r["seconds"] / 1e9 for r in received]})
        return

    if args.action == "upload":
        if not ray.get(task(gcs_absent).options(scheduling_strategy=on(pairs[0]["source_host"])).remote(bucket, prefix),
                       timeout=120):
            raise RuntimeError(f"{args.gcs} already holds objects; refusing to mix uploads")
        refs = [task(gcs_upload).options(scheduling_strategy=on(p["source_host"])).remote(
            bucket, prefix, args.source_scope, args.source, p["rank"],
            parts, args.workers, args.worker_type) for p in pairs]
        rows = ray.get(refs, timeout=args.timeout)
        wall = time.time() - started
        emit_measurements({**common, "ranks": rows, "bytes": sum(r["bytes"] for r in rows),
                           "files": sum(r["files"] for r in rows), "wall_s": wall,
                           "gbps": sum(r["bytes"] for r in rows) / wall / 1e9})
        return

    # fetch
    preps = ray.get([task(prepare_target).options(scheduling_strategy=on(p["target_host"])).remote(
        args.target_scope, args.target, p["rank"], parts) for p in pairs], timeout=120)
    transfer_started = time.time()
    senders = []
    if args.method in ("gcs", "fuse"):
        fn = gcs_fetch if args.method == "gcs" else fuse_fetch
        extra = (args.worker_type,) if args.method == "gcs" else ()
        rows = ray.get([task(fn).options(scheduling_strategy=on(p["target_host"])).remote(
            bucket, prefix, args.target_scope, args.target, p["rank"], parts, args.workers, *extra) for p in pairs],
            timeout=args.timeout)
    else:
        lister = task(source_files)
        listings = ray.get([lister.options(scheduling_strategy=on(p["source_host"])).remote(
            args.source_scope, args.source, p["rank"], parts) for p in pairs], timeout=120)
        receiver_cls = ray.remote(num_cpus=0, max_restarts=0)(Receiver)
        receivers = []
        for pair, files in zip(pairs, listings):
            plan = [[str(dest_path(args.target_scope, args.target, pair["rank"], f["part"], f["rel"])), f["bytes"]]
                    for f in files]
            receivers.append(receiver_cls.options(scheduling_strategy=on(pair["target_host"])).remote(
                plan, args.streams, args.timeout))
        ports = ray.get([r.port.remote() for r in receivers], timeout=120)
        transfer_started = time.time()
        runs = [r.run.remote() for r in receivers]
        sender = task(send_files)
        senders = ray.get([sender.options(scheduling_strategy=on(pair["source_host"])).remote(
            pair["target_host"], port, [[i, f["path"], f["bytes"]] for i, f in enumerate(files)],
            args.streams, args.timeout) for pair, port, files in zip(pairs, ports, listings)], timeout=args.timeout)
        received = ray.get(runs, timeout=args.timeout)
        rows = [{"rank": pair["rank"], **got, "by_part": {p: sum(f["bytes"] for f in files if f["part"] == p)
                                                           for p in parts}}
                for pair, files, got in zip(pairs, listings, received)]
    transfer_s = time.time() - transfer_started
    finalized_started = time.time()
    provenance = {"method": args.method, "source": args.source, "source_scope": args.source_scope,
                  "target": args.target, "run_scope": args.target_scope, "gcs": args.gcs}
    finals = ray.get([task(finalize).options(scheduling_strategy=on(p["target_host"])).remote(
        args.target_scope, args.target, p["rank"], parts,
        {**provenance, "source_host": p.get("source_host"), "source_rank": p["rank"]}) for p in pairs],
        timeout=args.timeout)
    finalize_s = time.time() - finalized_started
    for row, pair, prep, final in zip(rows, pairs, preps, finals):
        row.update(target_host=pair["target_host"], source_host=pair.get("source_host"),
                   before=prep["before"], published=final)
    total = sum(r["bytes"] for r in rows)
    ended = time.time()
    emit_measurements({**common, "method": args.method, "ranks": rows, "senders": senders, "bytes": total,
                       "files": sum(r["files"] for r in rows), "transfer_s": transfer_s,
                       "finalize_s": finalize_s, "wall_s": ended - started,
                       "started_at": started, "transfer_started_at": transfer_started,
                       "finalize_started_at": finalized_started, "ended_at": ended,
                       "transfer_gbps": total / transfer_s / 1e9 if transfer_s > 0 else None,
                       "per_host_gbps": [r["bytes"] / r["seconds"] / 1e9 for r in rows if r.get("seconds")]})


PROBE_ROOT = "/dev/shm/omp_peer_probe"


def probe_write(rank, sizes):
    directory = Path(PROBE_ROOT) / f"src-rank{int(rank)}"
    directory.mkdir(parents=True, exist_ok=True)
    block = bytes(range(256)) * (1 << 12)  # 1 MiB pattern
    paths = []
    for i, size in enumerate(sizes):
        path = directory / f"f{i}"
        with open(path, "wb") as handle:
            left = int(size)
            while left:
                left -= handle.write(block[:min(len(block), left)])
        paths.append(str(path))
    return paths


def probe_cleanup():
    shutil.rmtree(PROBE_ROOT, ignore_errors=True)
    return {"host": socket.gethostname(), "absent_after": not Path(PROBE_ROOT).exists()}


# ------------------------------------------------------------------ local submission

def placement_pairs(inventory, args):
    sys.path.insert(0, str(ROOT / "scripts"))
    from bank_relocate import congruent_pairs, placement
    if args.action == "upload":
        hosts, _, _ = placement(inventory, args.source)
        return [{"rank": rank, "source_host": ip} for rank, ip, _ in hosts]
    if (args.action == "fetch" and args.method == "peer") or args.action == "probe":
        return congruent_pairs(inventory, args.source, args.target)
    if args.action in ("fetch", "fetch-checkpoint", "ckpt-io"):
        hosts, _, _ = placement(inventory, args.target)
        return [{"rank": rank, "target_host": ip} for rank, ip, _ in hosts]
    return []


def remote_argv(args):
    """The remote driver's command line, with the rank pairs resolved from the local inventory."""
    from hetero.slice_inventory import SliceInventory
    pairs = []
    if args.action != "delete":
        data = json.loads(Path(args.inventory).read_text())
        inventory = SliceInventory.from_dict(data.get("measurements", {}).get("inventory", data))
        pairs = placement_pairs(inventory, args)
    argv = ["python3", "-u", "scripts/startup_state.py", args.action, "--remote", "--parts", args.parts,
            "--pairs", json.dumps(pairs), "--timeout", str(args.timeout), "--workers", str(args.workers),
            "--worker-type", args.worker_type, "--streams", str(args.streams),
            "--probe-bytes", str(args.probe_bytes), "--variants", args.variants]
    for flag, value in (("--method", args.method), ("--source", args.source), ("--target", args.target),
                        ("--source-scope", args.source_scope), ("--target-scope", args.target_scope),
                        ("--gcs", args.gcs)):
        if value:
            argv += [flag, value]
    return argv, pairs


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    argv, pairs = remote_argv(args)

    @stage(name=f"startup_state_{args.action}", order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        supervisor = state.get("supervisor")
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set("supervisor", supervisor)
        job = state.get("submission_id")
        if job is None:
            job = supervisor.submit_job(entrypoint=shlex.join(argv), runtime_env={
                "working_dir": str(ROOT), "excludes": [".venv*", ".git", ".jj", "__pycache__",
                "*.pyc", "results", "**/results/**", "reports", "data"],
                "env_vars": {"PYTHONUNBUFFERED": "1"}},
                metadata={"purpose": f"startup-state-{args.action}"}, name=f"startup-state-{args.action}")
            state.set("submission_id", job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                                                       stage_name=f"startup_state_{args.action}", state_store=state)
        measured = result["measurements"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"submission_id": job, "status": result["status"],
                                           "pairs": pairs, "measurements": measured}, indent=2) + "\n")
        if not measured or measured.get("ok") is not True:
            raise RuntimeError(f"startup-state {args.action} missing successful evidence: "
                               + str((measured or {}).get("error")))

    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("upload", "fetch", "delete", "probe", "fetch-checkpoint", "ckpt-io"))
    parser.add_argument("--method", choices=METHODS, help="fetch: gcs, fuse or peer; fetch-checkpoint: gcs or fuse")
    parser.add_argument("--variants", default=",".join(IO_VARIANTS),
                        help=f"ckpt-io: comma list, run in order, each a cold whole-checkpoint read on every "
                             f"target host at once (st-lazy, seq, par-N, gcs-N; default {','.join(IO_VARIANTS)})")
    parser.add_argument("--parts", default="snapshot,bank,xla", help="comma list of snapshot,bank,xla")
    parser.add_argument("--source", help="LAYOUT:INSTANCE whose state is shipped (upload, fetch --method peer)")
    parser.add_argument("--source-scope", help="run scope holding the source state (upload, fetch --method peer)")
    parser.add_argument("--target", help="LAYOUT:INSTANCE receiving the state (fetch)")
    parser.add_argument("--target-scope", help="run scope the fetch writes into (fetch)")
    parser.add_argument("--gcs", help="gs://<bucket>/<prefix> (upload, delete, fetch --method gcs|fuse)")
    parser.add_argument("--workers", type=int, default=32, help="GCS processes / fuse copy threads per host")
    parser.add_argument("--worker-type", choices=("process", "thread"), default="process",
                        help="google-cloud-storage transfer_manager worker type")
    parser.add_argument("--streams", type=int, default=8, help="peer: TCP streams per rank pair")
    parser.add_argument("--probe-bytes", type=int, default=4 << 30, help="probe: bytes per rank pair")
    parser.add_argument("--inventory", help="slice inventory JSON (hetero_ctl inventory output or its inventory object)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--print-remote-argv", action="store_true",
                        help="fetch: print the remote driver's argv as JSON and submit nothing (the deploy "
                             "job runs it next to the engine boot: hetero.cli deploy --state-delivery)")
    parser.add_argument("--pairs", help=argparse.SUPPRESS)
    parser.add_argument("--ray-url", default="http://127.0.0.1:8265")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    from hetero.run_scope import check_scope
    parse_parts(args.parts)
    need = {"upload": ("source", "source_scope", "gcs"), "delete": ("gcs",), "probe": ("source", "target"),
            "fetch": ("method", "target", "target_scope"),
            "fetch-checkpoint": ("gcs", "target", "target_scope"), "ckpt-io": ("gcs", "target")}[args.action]
    if args.action == "fetch-checkpoint" and args.method == "peer":
        parser.error("fetch-checkpoint --method is gcs or fuse")
    if args.action == "ckpt-io":
        for variant in args.variants.split(","):
            name, _, count = variant.removeprefix("st-lazy+").partition("-")
            base = variant.removeprefix("st-lazy+")
            if variant.startswith("st-lazy+") and not (base == "vpf" or (name == "par" and count.isdigit() and int(count) > 0)):
                parser.error(f"unknown ckpt-io variant {variant!r}")
            if not variant.startswith("st-lazy+") and variant not in ("st-lazy", "seq", "vpf", "fst", "fstb") \
                    and not (name in ("par", "gcs", "fst", "fstb", "sfd", "ofd") and count.isdigit() and int(count) > 0):
                parser.error(f"unknown ckpt-io variant {variant!r}")
    if args.action == "fetch":
        need += ("source", "source_scope") if args.method == "peer" else ("gcs",)
    missing = [n for n in need if not getattr(args, n)]
    if missing:
        parser.error(f"{args.action} needs " + ", ".join("--" + n.replace("_", "-") for n in missing))
    for scope in (args.source_scope, args.target_scope):
        if scope:
            check_scope(scope)
    if args.gcs:
        split_gcs(args.gcs)
    if args.remote:
        remote(args)
    elif args.print_remote_argv:
        if args.action != "fetch" or not args.inventory:
            parser.error("--print-remote-argv is for fetch, with --inventory")
        print(json.dumps(remote_argv(args)[0]))
    else:
        if (args.action != "delete" and not args.inventory) or args.output is None:
            parser.error("--inventory (except delete) and --output are required for supervised submission")
        submit(args)


if __name__ == "__main__":
    main()
