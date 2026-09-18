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
"""Deploy persistent KubeRay infrastructure without starting model workloads."""

import json
import os
import shlex
import sys
from typing import Optional

import fire
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from scripts.py import utils


def deploy_ray_cluster(
    config: str = "",
    values_file: str = "",
    namespace: Optional[str] = None,
    name: str = "tpu-ray-cluster",
    service_account: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    hf_secret_name: Optional[str] = None,
    image_repository: Optional[str] = None,
    image_tag: Optional[str] = None,
    ray_version: Optional[str] = None,
    accelerator_type: Optional[str] = None,
    topology: Optional[str] = None,
    node_count: Optional[int] = None,
    chips_per_pod: Optional[int] = None,
    yes: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
  """Install/upgrade only a RayCluster, never uninstall or clean up resources.

  config: Explicit infrastructure YAML path; reads namespace, service_account,
    gcs_bucket, and a single node_pools entry (topology/num_nodes). Multiple
    pools require topology and node_count in CLI flags or the values file.
    No config file or credentials are discovered implicitly.
  values_file: Helm values YAML. Precedence: explicit CLI flags > values file
    > infrastructure config > chart defaults. Namespace is not a chart value.
  name: Helm release name; default RayCluster is tpu-ray-cluster-vllm-tpu.
  gcs_bucket: Bucket name, not gs:// URI. Empty string disables GCS Fuse.
  hf_secret_name: Existing Secret with key HF_TOKEN; never an inline token.
  accelerator_type: tpu/v5e, tpu/v5p, tpu/v6e, or tpu/v7.
  image_repository: Image containing the selected Ray version and TPU runtime.
    Operator must verify compatibility; this command does not load a model.
  ray_version: Ray version installed in the image (default from chart: 2.56.0).
  dry_run: Print the command offline; no subprocesses or dependency checks.
  yes: Confirm execution without an interactive prompt.
  """
  deployment_config = {}
  values = {}
  try:
    for path, target in ((config, deployment_config), (values_file, values)):
      if path:
        with open(path, encoding="utf-8") as stream:
          data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
          raise ValueError(f"{path} must contain a YAML mapping")
        target.update(data)
  except (OSError, ValueError, yaml.YAMLError) as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return False

  namespace = namespace if namespace is not None else deployment_config.get(
      "namespace", "llm-d-optimized-baseline"
  )
  overrides = {}
  if "serviceAccountName" not in values and deployment_config.get("service_account"):
    overrides["serviceAccountName"] = deployment_config["service_account"]
  storage = values.get("storage", {})
  if not isinstance(storage, dict):
    print("Error: values storage must be a mapping", file=sys.stderr)
    return False
  if deployment_config.get("gcs_bucket"):
    if "type" not in storage:
      overrides["storage.type"] = "gcsFuse"
    if "gcsBucket" not in storage:
      overrides["storage.gcsBucket"] = deployment_config["gcs_bucket"]
  tpu_values = values.get("tpu", {})
  pools = deployment_config.get("node_pools", [])
  if not isinstance(tpu_values, dict) or not isinstance(pools, list):
    print("Error: values tpu must be a mapping and config node_pools a list", file=sys.stderr)
    return False
  if len(pools) > 1 and (
      (topology is None and "topology" not in tpu_values)
      or (node_count is None and "nodeCount" not in tpu_values)
  ):
    print("Error: multiple node_pools require explicit topology and node_count", file=sys.stderr)
    return False
  if len(pools) == 1:
    if not isinstance(pools[0], dict):
      print("Error: node_pools entries must be mappings", file=sys.stderr)
      return False
    for value_key, config_key in (("topology", "topology"), ("nodeCount", "num_nodes")):
      if value_key not in tpu_values and config_key in pools[0]:
        overrides[f"tpu.{value_key}"] = pools[0][config_key]
  for key, value in (
      ("serviceAccountName", service_account),
      ("hfSecretName", hf_secret_name),
      ("image.repository", image_repository),
      ("image.tag", image_tag),
      ("rayVersion", ray_version),
      ("tpu.topology", topology),
      ("tpu.nodeCount", node_count),
      ("tpu.chipsPerPod", chips_per_pod),
  ):
    if value is not None:
      overrides[key] = value
  if accelerator_type is not None:
    if accelerator_type not in ("tpu/v5e", "tpu/v5p", "tpu/v6e", "tpu/v7"):
      print("Error: accelerator_type must be tpu/v5e, tpu/v5p, tpu/v6e, or tpu/v7", file=sys.stderr)
      return False
    overrides["tpu.generation"] = accelerator_type.split("/")[1]
  if gcs_bucket is not None:
    overrides["storage.type"] = "gcsFuse" if gcs_bucket else "none"
    overrides["storage.gcsBucket"] = gcs_bucket

  chart_path = os.path.join(project_root, "k8s", "vllm-tpu-chart")
  cmd = ["helm", "upgrade", "--install", name, chart_path,
         "--namespace", namespace, "--create-namespace"]
  if values_file:
    cmd.extend(["--values", values_file])
  for key, value in overrides.items():
    # JSON preserves types and prevents commas/backslashes becoming extra sets.
    cmd.extend(["--set-json", f"{key}={json.dumps(value)}"])

  if dry_run:
    return utils.run_command(cmd, dry_run=True, verbose=verbose)
  print(f"Helm command: {shlex.join(cmd)}")
  if not yes:
    try:
      if input("Install/upgrade this RayCluster? (y/n) ").strip().lower() not in ("y", "yes"):
        print("Deployment cancelled.")
        return False
    except (KeyboardInterrupt, EOFError):
      print("\nDeployment cancelled.")
      return False
  return utils.run_command(cmd, dry_run=False, verbose=verbose)


if __name__ == "__main__":
  if fire.Fire(deploy_ray_cluster) is False:
    sys.exit(1)
