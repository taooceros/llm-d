#!/usr/bin/env python3
"""Provision missing GKE infrastructure; never replace existing TPU pools."""

from typing import Any, Dict

import fire
import yaml

try:
    from scripts.py import utils
except ImportError:
    import utils


def create_node_pool(
    np: Dict[str, Any],
    cluster_name: str,
    zone: str,
    project: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """Create a missing TPU pool after the caller has obtained confirmation."""
    reservation = np.get("reservation")
    reservation_affinity = np.get("reservation_affinity") or (
        "specific" if reservation else "none"
    )
    args = [
        "gcloud", "beta", "container", "node-pools", "create", np["name"],
        f"--cluster={cluster_name}", f"--project={project}", f"--zone={zone}",
        f"--node-locations={zone}", f"--machine-type={np['machine_type']}",
        f"--reservation-affinity={reservation_affinity}",
        "--scopes=https://www.googleapis.com/auth/cloud-platform", "--quiet",
    ]
    if np.get("topology"):
        args.append(f"--tpu-topology={np['topology']}")
    if np.get("num_nodes") is not None:
        args.append(f"--num-nodes={np['num_nodes']}")
    if reservation_affinity == "specific":
        args.append(f"--reservation={reservation}")
    if str(np.get("spot", False)).lower() == "true":
        args.append("--spot")
    if np.get("accelerator_network_profile"):
        args.append(f"--accelerator-network-profile={np['accelerator_network_profile']}")
    labels = np.get("node_labels")
    if labels:
        if isinstance(labels, dict):
            labels = ",".join(f"{key}={value}" for key, value in labels.items())
        args.append(f"--node-labels={labels}")
    return utils.run_command(args, dry_run=dry_run, verbose=verbose)


def node_pool_config_matches(current: Dict[str, Any], desired: Dict[str, Any]) -> bool:
    """Compare the selected TPU pool settings without changing existing pools."""
    config = current.get("config", {})
    if config.get("machineType", "").split("/")[-1] != desired["machine_type"].split("/")[-1]:
        return False
    if config.get("spot", False) != (str(desired.get("spot", False)).lower() == "true"):
        return False
    if desired.get("topology") and current.get("placementPolicy", {}).get("tpuTopology") != desired["topology"]:
        return False
    if desired.get("num_nodes") is not None and current.get("initialNodeCount") != desired["num_nodes"]:
        return False
    if current.get("autoscaling", {}).get("enabled", False):
        return False
    if desired.get("accelerator_network_profile") and current.get("networkConfig", {}).get("acceleratorNetworkProfile") != desired["accelerator_network_profile"]:
        return False
    labels = desired.get("node_labels")
    if labels:
        if not isinstance(labels, dict):
            labels = dict(item.split("=", 1) for item in labels.split(","))
        if any(config.get("labels", {}).get(key) != str(value) for key, value in labels.items()):
            return False
    reservation = desired.get("reservation")
    desired_affinity = desired.get("reservation_affinity") or (
        "specific" if reservation else "none"
    )
    affinity = config.get("reservationAffinity", {})
    current_affinity = {
        "NO_RESERVATION": "none", "ANY_RESERVATION": "any",
        "SPECIFIC_RESERVATION": "specific",
    }.get(affinity.get("consumeReservationType"))
    if current_affinity != desired_affinity:
        return False
    if desired_affinity == "specific" and (
        affinity.get("key") != "compute.googleapis.com/reservation-name"
        or reservation not in affinity.get("values", [])
    ):
        return False
    return True


def main(
    config: str = "config.yaml",
    dry_run: bool = False,
    verbose: bool = True,
    yes: bool = False,
):
    """Create missing GKE cluster/TPU pools and acquire cluster credentials.

    Existing pools must be RUNNING and match config; drift requires deliberate
    operator reconciliation outside this script. --yes authorizes mutations.
    Dry runs render a conditional plan with no subprocess or network calls.
    """
    with open(config) as stream:
        data = yaml.safe_load(stream) or {}
    if "hf_token" in data:
        raise SystemExit("Plaintext token configuration is unsupported; use an existing Kubernetes Secret at deployment.")
    project = data.get("project_id")
    region = data.get("region")
    zone = data.get("zone")
    cluster_name = data.get("cluster_name")
    network = data.get("network")
    subnetwork = data.get("subnetwork")
    if not project or project == "your-gcp-project-id":
        raise SystemExit("project_id must be set in config")
    if not all((region, zone, cluster_name, network, subnetwork)):
        raise SystemExit("region, zone, cluster_name, network, and subnetwork must be set in config")
    cluster_name = cluster_name.replace("{region}", region).replace("{zone}", zone).replace("{project_id}", project)
    pools = data.get("node_pools", [])
    if isinstance(pools, dict):
        pools = [pools]
    if not isinstance(pools, list) or not pools:
        raise SystemExit("At least one node pool must be specified in config")
    names = set()
    for pool in pools:
        if not isinstance(pool, dict) or not pool.get("name") or not pool.get("machine_type"):
            raise SystemExit("Every node pool needs a name and machine_type")
        if pool["name"] in names:
            raise SystemExit(f"Duplicate node pool name: {pool['name']}")
        names.add(pool["name"])
        affinity = pool.get("reservation_affinity")
        reservation = pool.get("reservation")
        if affinity and affinity not in ("none", "any", "specific"):
            raise SystemExit("reservation_affinity must be none, any, or specific")
        if affinity == "specific" and not reservation:
            raise SystemExit("specific reservation_affinity requires a reservation")
        if affinity in ("none", "any") and reservation:
            raise SystemExit("reservation requires specific reservation_affinity")

    print(f"Cluster: {project}/{zone}/{cluster_name}; network: {network}/{subnetwork}")
    if not dry_run and not yes:
        try:
            confirmed = input(
                "Create missing cluster/node pools and update local kubeconfig? [y/N] "
            ).strip().lower() in ("y", "yes")
        except EOFError:
            confirmed = False
        if not confirmed:
            raise SystemExit("Cancelled; no changes made. Use --yes for noninteractive execution.")
    if dry_run:
        print("[DRY RUN] Conditional plan: create only if absent; reject existing cluster/pool drift.")

    scope = [f"--project={project}", f"--zone={zone}"]
    clusters = utils.list_resources(
        ["gcloud", "container", "clusters", "list", *scope], dry_run, verbose
    )
    cluster = next((item for item in clusters if item.get("name") == cluster_name), None)
    existing_pools = []
    if cluster is not None:
        if cluster.get("status") != "RUNNING":
            raise SystemExit(f"Cluster {cluster_name} is not RUNNING; refusing changes.")
        for field, expected in (("network", network), ("subnetwork", subnetwork)):
            if cluster.get(field, "").split("/")[-1] != expected.split("/")[-1]:
                raise SystemExit(f"Cluster {cluster_name} has incompatible {field}; refusing changes.")
        print(f"[SKIP] Existing cluster {cluster_name}")
    if cluster is not None or dry_run:
        existing_pools = utils.list_resources(
            ["gcloud", "container", "node-pools", "list", f"--cluster={cluster_name}", *scope],
            dry_run, verbose,
        )
    by_name = {pool.get("name"): pool for pool in existing_pools}
    # Validate all existing requested pools before creating any missing pool.
    for pool in pools:
        current = by_name.get(pool["name"])
        if current is None:
            continue
        if current.get("status") != "RUNNING" or not node_pool_config_matches(current, pool):
            raise SystemExit(
                f"Node pool {pool['name']} has drift or is not RUNNING "
                f"(status={current.get('status')}); refusing to delete, recreate, or change it. "
                "Inspect and reconcile it explicitly before retrying."
            )
        if current.get("locations") != [zone]:
            raise SystemExit(f"Node pool {pool['name']} has incompatible locations; refusing changes.")

    if cluster is None:
        args = [
            "gcloud", "container", "clusters", "create", cluster_name, *scope,
            f"--num-nodes={data.get('cpu_num_nodes', 1)}",
            f"--machine-type={data.get('cpu_machine_type', 'e2-standard-4')}",
            "--enable-ip-alias", f"--workload-pool={project}.svc.id.goog",
            f"--network={network}", f"--subnetwork={subnetwork}", "--quiet",
        ]
        for key, flag in (("release_channel", "release-channel"), ("addons", "addons"), ("gateway_api", "gateway-api")):
            if data.get(key):
                args.append(f"--{flag}={data[key]}")
        if data.get("enable_dataplane_v2"):
            args.extend(["--enable-dataplane-v2", "--enable-dataplane-v2-metrics"])
        if data.get("enable_managed_prometheus"):
            args.append("--enable-managed-prometheus")
        if not utils.run_command(args, dry_run, verbose):
            raise SystemExit(1)
    for pool in pools:
        if pool["name"] in by_name:
            print(f"[SKIP] Matching RUNNING node pool {pool['name']}")
        elif not create_node_pool(pool, cluster_name, zone, project, dry_run, verbose):
            raise SystemExit(1)

    if not utils.run_command(
        ["gcloud", "container", "clusters", "get-credentials", cluster_name, *scope],
        dry_run, verbose,
    ):
        raise SystemExit(1)
    print("Cluster plan complete." if dry_run else "Cluster setup complete.")


if __name__ == "__main__":
    fire.Fire(main)
