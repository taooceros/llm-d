#!/usr/bin/env python3
"""Assemble the five research forks without rewriting their Python imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

REPOS = ("vllm", "tpu-inference", "llm-d-router", "llm-d", "llm-d-async")
SUBTREE = Path("experimental/heterogeneous-serving")


def source_files(forks_root: Path) -> tuple[list[tuple[str, Path, dict]], str]:
    """Validate every source before creating any output."""
    files = []
    owners: dict[str, str] = {}
    commits = set()
    for repo in REPOS:
        root = (forks_root / repo / SUBTREE).resolve()
        manifest = json.loads((root / "migration.json").read_text())
        if manifest["format"] != 1 or manifest["owner"] != repo:
            raise ValueError(f"Invalid source manifest: {repo}")
        commits.add(manifest["source_commit"])
        for item in manifest["workspace_files"]:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or str(relative) != item["path"]:
                raise ValueError(f"Unsafe source path: {item['path']}")
            source = root / relative
            if source.is_symlink() or not source.resolve().is_relative_to(root):
                raise ValueError(f"Source escapes its fork: {source}")
            data = source.read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Source differs from migration manifest: {repo}/{relative}")
            if str(relative) in owners:
                raise ValueError(f"Duplicate ownership: {relative}: {owners[str(relative)]}, {repo}")
            owners[str(relative)] = repo
            files.append((repo, source, item))
    if len(commits) != 1:
        raise ValueError("Forks do not describe the same original research snapshot")
    return files, commits.pop()


def assemble(forks_root: Path, output: Path) -> dict:
    files, original_commit = source_files(forks_root)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a fresh workspace: {output}")
    if output.is_relative_to(forks_root.resolve()):
        raise ValueError("Generated workspace must be outside the fork checkout directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "format": 1,
        "original_source_commit": original_commit,
        "source_files": [{"owner": repo, **item} for repo, _, item in files],
        "qualification": "Historical evidence applies to its frozen source/native builds, not automatically to newer upstream bases.",
    }
    with tempfile.TemporaryDirectory(prefix=".hetero-assembly-", dir=output.parent) as temporary:
        stage = Path(temporary) / "workspace"
        stage.mkdir()
        for _, source, item in files:
            target = stage / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Source changed during assembly: {source}")
        (stage / "workspace-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        stage.rename(output)
    return {"workspace": str(output), "copied_files": len(files), "original_source_commit": original_commit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forks-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify all source manifests without creating a workspace")
    args = parser.parse_args()
    if args.check:
        files, commit = source_files(args.forks_root)
        result = {"verified_files": len(files), "original_source_commit": commit}
    else:
        if args.output is None:
            parser.error("--output is required unless --check is used")
        result = assemble(args.forks_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
