"""Command execution helpers for the selected infrastructure scripts."""

import json
import shlex
import subprocess
import sys


def run_command(cmd, dry_run=False, verbose=True):
    """Run an argv command; dry runs never start a subprocess."""
    if dry_run:
        print(f"[DRY RUN] {shlex.join(cmd)}")
        return True
    if verbose:
        print(f"Running: {shlex.join(cmd)}")
    try:
        subprocess.run(
            cmd, check=True, text=True,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
        )
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        if getattr(exc, "stderr", None):
            print(exc.stderr, file=sys.stderr)
        return False


def list_resources(cmd, dry_run=False, verbose=True):
    """List JSON resources, aborting on probe errors instead of assuming absence.

    Dry runs return an empty list to render conditional create commands only.
    """
    cmd = [*cmd, "--format=json"]
    if dry_run:
        run_command(cmd, dry_run=True)
        return []
    if verbose:
        print(f"Inspecting: {shlex.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        resources = json.loads(result.stdout)
        if not isinstance(resources, list) or any(
            not isinstance(resource, dict) for resource in resources
        ):
            raise ValueError("Expected a JSON resource list")
        return resources
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Resource inspection failed; refusing to create: {exc}", file=sys.stderr)
        if getattr(exc, "stderr", None):
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(1) from exc
