#!/usr/bin/env python3
"""Validate and restore research release assets using a trusted checked-in index.

Checks SHA256 of inventories, archives and every uncompressed member before any
extraction. Only canonical relative regular-file members are accepted. Existing
files are never overwritten; --merge permits existing directories/source files.
Use a trusted index from the source checkout, not an untrusted downloaded index.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name or '\0' in name:
        raise ValueError(f'Unsafe member name: {name!r}')
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in {'.', '..', ''} for part in name.split('/')):
        raise ValueError(f'Unsafe member name: {name!r}')
    if ':' in p.parts[0] or str(p) != name:
        raise ValueError(f'Noncanonical member name: {name!r}')
    return p.parts


def local_file(base, entry):
    parts = safe_name(entry['name'])
    if len(parts) != 1:
        raise ValueError('Asset and inventory names must be basenames')
    path = base / entry['name']
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Not a regular input file: {path}')
    if path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
        raise ValueError(f'SHA256 or size mismatch: {path}')
    return path


def check_destination(destination, paths, merge):
    # abspath normalizes lexical spelling without following symlinks.
    destination = Path(os.path.abspath(destination))
    for parent in reversed([destination, *destination.parents]):
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError(f'Unsafe destination ancestor: {parent}')
    if destination.exists() and not merge and any(destination.iterdir()):
        raise ValueError('Destination must be empty; use --merge to overlay without overwrite')
    for name in paths:
        target = destination.joinpath(*safe_name(name))
        if target.exists() or target.is_symlink():
            raise ValueError(f'Refusing existing destination file: {target}')
        for parent in target.parents:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(f'Unsafe destination ancestor: {parent}')
            if parent == destination:
                break
    return destination


def open_directory_chain(path):
    """Open/create directories without following symlinks, including ancestors."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open('/', flags)
    try:
        for component in Path(os.path.abspath(path)).parts[1:]:
            try:
                os.mkdir(component, mode=0o755, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def write_member(root_fd, name, stream, expected):
    parts = safe_name(name)
    parent = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, mode=0o755, dir_fd=parent)
            except FileExistsError:
                pass
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=parent)
        digest, size = hashlib.sha256(), 0
        with os.fdopen(fd, 'wb') as output:
            while block := stream.read(1024 * 1024):
                digest.update(block)
                size += len(block)
                output.write(block)
        if size != expected['size'] or digest.hexdigest() != expected['sha256']:
            os.unlink(parts[-1], dir_fd=parent)
            raise ValueError(f'Member changed after validation: {name}')
    finally:
        os.close(parent)


def restore(index_path, assets_dir, destination=None, merge=False, verify_only=False):
    index = json.loads(index_path.read_text())
    if index['format'] != 1:
        raise ValueError('Unsupported index format')
    records = {}
    for inventory in index['inventories']:
        path = local_file(index_path.parent, inventory)
        for record in json.loads(path.read_text()):
            safe_name(record['path'])
            if record['path'] in records:
                raise ValueError(f'Duplicate inventory path: {record["path"]}')
            records[record['path']] = record
    # Reject file/parent collisions before any output writes.
    for name in records:
        if any(str(parent) in records for parent in PurePosixPath(name).parents):
            raise ValueError(f'File/parent inventory collision: {name}')
    asset_paths = {}
    for asset in index['assets']:
        name = asset['name']
        if name in asset_paths:
            raise ValueError(f'Duplicate asset: {name}')
        asset_paths[name] = local_file(assets_dir, asset)
    if any(record['asset'] not in asset_paths for record in records.values()):
        raise ValueError('Inventory refers to an undeclared asset')
    if len(records) != index['files']:
        raise ValueError('Inventory file count mismatch')
    seen, total = set(), 0
    for asset_name, path in asset_paths.items():
        with tarfile.open(path, 'r|gz') as archive:
            for member in archive:
                safe_name(member.name)
                if not member.isreg() or member.issym() or member.islnk() or member.linkname:
                    raise ValueError(f'Non-regular archive member: {member.name}')
                expected = records.get(member.name)
                if member.name in seen or expected is None or expected['asset'] != asset_name:
                    raise ValueError(f'Unexpected/duplicate archive member: {member.name}')
                if member.size != expected['size']:
                    raise ValueError(f'Member size mismatch: {member.name}')
                with archive.extractfile(member) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != expected['sha256']:
                    raise ValueError(f'Member SHA256 mismatch: {member.name}')
                seen.add(member.name)
                total += member.size
    if seen != records.keys() or total != index['uncompressed_bytes']:
        raise ValueError('Archive inventory coverage mismatch')
    if destination is not None:
        destination = check_destination(destination, records, merge)
    elif not verify_only:
        raise ValueError('--destination is required for extraction')
    if not verify_only:
        root_fd = open_directory_chain(destination)
        try:
            for path in asset_paths.values():
                with tarfile.open(path, 'r|gz') as archive:
                    for member in archive:
                        if not member.isreg() or member.linkname or member.name not in records:
                            raise ValueError('Archive changed after validation')
                        with archive.extractfile(member) as stream:
                            write_member(root_fd, member.name, stream, records[member.name])
        finally:
            os.close(root_fd)
    return {'verified': True, 'extracted': not verify_only, 'files': len(seen),
            'bytes': total, 'assets': len(asset_paths), 'index_sha256': sha256(index_path),
            'destination': str(destination) if destination else None,
            'merge_without_overwrite': merge}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=Path(__file__).resolve().parent / 'asset-index.json')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(restore(args.index, args.assets, args.destination, args.merge, args.verify_only), indent=2))


if __name__ == '__main__':
    main()
