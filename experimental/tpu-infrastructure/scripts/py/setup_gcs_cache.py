#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create an explicit GCS bucket and zonal Anywhere Cache, with optional IAM."""

import os
from typing import Optional

import fire
import yaml

try:
    from scripts.py import utils
except ImportError:
    import utils


def setup_gcs_cache(
    config: str = "config.yaml",
    project_id: Optional[str] = None,
    region: Optional[str] = None,
    zone: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    ttl: str = "1d",
    yes: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
    iam_member: Optional[str] = None,
    iam_role: str = "roles/storage.objectViewer",
) -> None:
    """Create missing resources after confirmation; never infer an IAM member.

    iam_member overrides config.workload_identity_member. Without either, no
    IAM policy is changed. Bucket names may contain {region} and {zone}.
    Dry runs print conditional commands and never inspect cloud state.
    """
    data = {}
    if os.path.exists(config):
        with open(config) as stream:
            data = yaml.safe_load(stream) or {}
    project_id = project_id or data.get("project_id")
    region = region or data.get("region")
    zone = zone or data.get("zone")
    gcs_bucket = gcs_bucket or data.get("gcs_bucket")
    iam_member = iam_member or data.get("workload_identity_member")
    if not all((project_id, region, zone, gcs_bucket)):
        raise SystemExit("project_id, region, zone, and an explicit gcs_bucket are required")
    gcs_bucket = gcs_bucket.replace("{region}", region).replace("{zone}", zone)
    if gcs_bucket.startswith("gs://") or "/" in gcs_bucket:
        raise SystemExit("gcs_bucket must be a bucket name, not a URL or path")
    if iam_member and ("*" in iam_member or iam_member in ("allUsers", "allAuthenticatedUsers")):
        raise SystemExit("IAM member must identify an explicit principal, not a whole pool or public wildcard")

    if not dry_run and not yes:
        try:
            confirmed = input(
                f"Create missing bucket/cache resources in {project_id}"
                f"{' and apply the explicit IAM binding' if iam_member else ''}? [y/N] "
            ).strip().lower() in ("y", "yes")
        except EOFError:
            confirmed = False
        if not confirmed:
            raise SystemExit("Cancelled; no changes made. Use --yes for noninteractive execution.")
    if dry_run:
        print("[DRY RUN] Conditional plan: create only if absent; no cloud state queried.")

    scope = [f"--project={project_id}"]
    bucket_url = f"gs://{gcs_bucket}"
    buckets = utils.list_resources(
        ["gcloud", "storage", "buckets", "list", *scope, "--raw"], dry_run, verbose
    )
    bucket = next((item for item in buckets if item.get("name") == gcs_bucket), None)
    if bucket is not None:
        if bucket.get("location", "").lower() != region.lower():
            raise SystemExit(f"Bucket {gcs_bucket} has a different location; refusing changes.")
        print(f"[SKIP] Matching bucket {bucket_url}")
    elif not utils.run_command(
        ["gcloud", "storage", "buckets", "create", bucket_url, *scope,
         f"--location={region}", "--uniform-bucket-level-access"], dry_run, verbose
    ):
        raise SystemExit(1)

    caches = utils.list_resources(
        ["gcloud", "storage", "buckets", "anywhere-caches", "list", bucket_url,
         *scope, "--raw"], dry_run, verbose
    )
    if any(item.get("zone") == zone for item in caches):
        print(f"[SKIP] Anywhere Cache for {bucket_url} in {zone}")
    elif not utils.run_command(
        ["gcloud", "storage", "buckets", "anywhere-caches", "create", bucket_url,
         zone, *scope, f"--ttl={ttl}"], dry_run, verbose
    ):
        raise SystemExit(1)

    if iam_member:
        if not utils.run_command(
            ["gcloud", "storage", "buckets", "add-iam-policy-binding", bucket_url,
             *scope, f"--member={iam_member}", f"--role={iam_role}"], dry_run, verbose
        ):
            raise SystemExit(1)
    else:
        print("[SKIP] IAM unchanged; no explicit member supplied.")
    print("Bucket/cache plan complete." if dry_run else "Bucket/cache setup complete.")


if __name__ == "__main__":
    fire.Fire(setup_gcs_cache)
