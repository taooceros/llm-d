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
"""Create or verify a custom VPC, exact subnet, and CIDR-scoped firewall."""

import ipaddress
import os
from typing import Optional

import fire
import yaml

try:
    from scripts.py import utils
except ImportError:
    import utils


def setup_network(
    config: str = "config.yaml",
    project_id: Optional[str] = None,
    region: Optional[str] = None,
    network: Optional[str] = None,
    subnetwork: Optional[str] = None,
    ip_range: Optional[str] = None,
    mtu: Optional[int] = None,
    yes: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Reuse matching resources, or create missing ones after confirmation.

    ip_range and mtu default to config values, then 192.168.100.0/24 and 8896.
    Dry runs render conditional creates without inspecting cloud state.
    """
    data = {}
    if os.path.exists(config):
        with open(config) as stream:
            data = yaml.safe_load(stream) or {}
    project_id = project_id or data.get("project_id")
    region = region or data.get("region")
    network = network or data.get("network")
    subnetwork = subnetwork or data.get("subnetwork")
    ip_range = ip_range or data.get("ip_range", "192.168.100.0/24")
    mtu = mtu if mtu is not None else data.get("mtu", 8896)
    if not all((project_id, region, network, subnetwork)):
        raise SystemExit("project_id, region, network, and subnetwork are required")
    ipaddress.IPv4Network(ip_range)

    if not dry_run and not yes:
        try:
            confirmed = input(
                f"Create missing network resources in {project_id}/{region}? [y/N] "
            ).strip().lower() in ("y", "yes")
        except EOFError:
            confirmed = False
        if not confirmed:
            raise SystemExit("Cancelled; no changes made. Use --yes for noninteractive execution.")
    if dry_run:
        print("[DRY RUN] Conditional plan: create only if absent; reject existing resource drift.")

    scope = [f"--project={project_id}"]
    networks = utils.list_resources(
        ["gcloud", "compute", "networks", "list", *scope], dry_run, verbose
    )
    subnets = utils.list_resources(
        ["gcloud", "compute", "networks", "subnets", "list", *scope,
         f"--regions={region}"], dry_run, verbose
    )
    firewalls = utils.list_resources(
        ["gcloud", "compute", "firewall-rules", "list", *scope], dry_run, verbose
    )
    existing_network = next((item for item in networks if item.get("name") == network), None)
    existing_subnet = next((item for item in subnets if item.get("name") == subnetwork), None)
    firewall_name = f"{network}-allow-internal"
    existing_firewall = next((item for item in firewalls if item.get("name") == firewall_name), None)
    network_ref = f"projects/{project_id}/global/networks/{network}"

    def same_network(value):
        return value == network_ref or value.endswith("/" + network_ref)

    if existing_network is not None and (
        existing_network.get("mtu") != mtu
        or existing_network.get("autoCreateSubnetworks") is not False
    ):
        raise SystemExit(f"Network {network} has incompatible MTU or subnet mode; refusing changes.")
    if existing_subnet is not None and (
        not same_network(existing_subnet.get("network", ""))
        or existing_subnet.get("ipCidrRange") != ip_range
    ):
        raise SystemExit(f"Subnet {subnetwork} has incompatible network or CIDR; refusing changes.")
    if existing_firewall is not None and (
        not same_network(existing_firewall.get("network", ""))
        or existing_firewall.get("sourceRanges") != [ip_range]
        or existing_firewall.get("direction") != "INGRESS"
        or existing_firewall.get("disabled", False)
        or existing_firewall.get("allowed") != [{"IPProtocol": "all"}]
        or existing_firewall.get("denied")
        or existing_firewall.get("sourceTags")
        or existing_firewall.get("sourceServiceAccounts")
        or existing_firewall.get("targetTags")
        or existing_firewall.get("targetServiceAccounts")
    ):
        raise SystemExit(f"Firewall {firewall_name} is not the configured CIDR-scoped rule; refusing changes.")

    planned = [
        (existing_network, ["gcloud", "compute", "networks", "create", network,
                            *scope, "--subnet-mode=custom", f"--mtu={mtu}"]),
        (existing_subnet, ["gcloud", "compute", "networks", "subnets", "create", subnetwork,
                           *scope, f"--network={network}", f"--region={region}", f"--range={ip_range}"]),
        (existing_firewall, ["gcloud", "compute", "firewall-rules", "create", firewall_name,
                             *scope, f"--network={network}", "--allow=all",
                             f"--source-ranges={ip_range}", "--direction=INGRESS"]),
    ]
    for existing, command in planned:
        if existing is not None:
            print(f"[SKIP] Matching resource {existing['name']}")
        elif not utils.run_command(command, dry_run=dry_run, verbose=verbose):
            raise SystemExit(1)
    print("Network plan complete." if dry_run else "Network setup complete.")


if __name__ == "__main__":
    fire.Fire(setup_network)
