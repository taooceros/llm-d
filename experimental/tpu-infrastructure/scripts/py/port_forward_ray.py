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
"""Port-forward one explicitly named KubeRay cluster's head dashboard."""

import os
import sys

import fire

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from scripts.py import utils


def port_forward_ray(
    cluster_name: str,
    namespace: str = "llm-d-optimized-baseline",
    local_port: int = 8265,
    address: str = "127.0.0.1",
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
  """Forward the Ray dashboard, not model traffic; never discover another cluster.

  cluster_name: RayCluster metadata.name, not necessarily its Helm release name.
    Default chart release tpu-ray-cluster renders tpu-ray-cluster-vllm-tpu.
  namespace: Namespace containing the RayCluster and its head Service.
  local_port: Local TCP port forwarded to the head dashboard on port 8265.
  address: Local bind address (loopback by default).
  dry_run: Print the command with no subprocesses or cluster access.
  """
  if not cluster_name or not isinstance(local_port, int) or not 1 <= local_port <= 65535:
    print("Error: cluster_name and a local_port between 1 and 65535 are required", file=sys.stderr)
    return False
  cmd = ["kubectl", "port-forward", "--namespace", namespace,
         "--address", address, f"service/{cluster_name}-head-svc",
         f"{local_port}:8265"]
  return utils.run_command(cmd, dry_run=dry_run, verbose=verbose)


if __name__ == "__main__":
  if fire.Fire(port_forward_ray) is False:
    sys.exit(1)
