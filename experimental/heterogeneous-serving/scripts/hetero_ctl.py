#!/usr/bin/env python3
"""Local RCM control plane for the heterogeneous serving fabric.

Submits ``hetero.cli`` commands to the existing Ray cluster through the RCM
``RayJobSupervisor`` and saves the structured evidence payload.

    rcm_exec run scripts/hetero_ctl.py -- inventory --output results/hetero/inventory.json
    rcm_exec run scripts/hetero_ctl.py -- plan --layout M --inventory results/hetero/inventory.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omp_rcm import StageRunner, stage  # noqa: E402
from omp_rcm.ray_job import RayJobSupervisor  # noqa: E402

RUNTIME_EXCLUDES = [
    ".venv*",
    ".git",
    ".jj",
    "formalization",
    "skaffold",
    "__pycache__",
    "*.pyc",
    "docs",
    "results",
    "experiments",
    "reports",
    "**/results/**",
    "**/reports/**",
    "data/workloads/sharegpt_large.json",
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", help="hetero.cli command (inventory, plan, ...)")
parser.add_argument("--ray-url", default="http://127.0.0.1:8265")
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--timeout", type=float, default=2400.0)
parser.add_argument("--layout", default="M")
parser.add_argument("--layout-file")
parser.add_argument("--inventory", help="local slice inventory JSON uploaded with the job")
parser.add_argument("--protocol", help="local frozen qualification/experiment protocol JSON")
parser.add_argument("--qualification-report", help="local passed native continuation report JSON for mechanism-cost")
parser.add_argument("--evaluation-protocol", help="qualified source-frozen evaluation JSON for mechanism-cost")
parser.add_argument("--force", action="store_true")
parser.add_argument("--namespace", default="llm-d-optimized-baseline")
parser.add_argument("--head-pod", default="tpu-ray-cluster-vllm-tpu-head-llg9k")
parser.add_argument("--data-dir", default="/tmp/omp_hetero/data")
parser.add_argument(
    "--include-data",
    action="store_true",
    help="upload the full ShareGPT catalogue and historical manifest (manifest build only)",
)
args, extra = parser.parse_known_args()

input_env = {}
input_files = [
    (args.inventory, "OMP_HETERO_INVENTORY_JSON"),
    (args.layout_file, "OMP_HETERO_LAYOUT_JSON"),
    (args.protocol, "OMP_HETERO_PROTOCOL_JSON"),
    (args.qualification_report, "OMP_HETERO_QUALIFICATION_REPORT_JSON"),
    (args.evaluation_protocol, "OMP_HETERO_EVALUATION_PROTOCOL_JSON"),
]
if args.command == "mechanism-cost":
    input_files.append((ROOT / "results/hetero/continuation_qualification_protocol.json",
                        "OMP_HETERO_FIDELITY_PROTOCOL_JSON"))
for filename, key in input_files:
    if filename:
        raw = pathlib.Path(filename).read_bytes()
        json.loads(raw)
        # Environment transport preserves UTF-8 bytes, including CRLF and BOM.
        input_env[key] = raw.decode("utf-8")

remote_args: list[str] = [args.command, "--layout", args.layout]
if args.layout_file:
    remote_args += ["--layout-file", args.layout_file]
if args.inventory:
    remote_args += ["--inventory-file", args.inventory]
if args.force:
    remote_args += ["--force"]
if extra[:1] == ["--"]:
    extra = extra[1:]
remote_args += extra

if args.include_data:
    RUNTIME_EXCLUDES = [
        e
        for e in RUNTIME_EXCLUDES
        if e not in ("data/workloads/sharegpt_large.json", "results", "**/results/**")
    ] + ["results/hetero", "results/decode_allocation_20260913", "results/scheduling_tracks_20260914"]


# Results are normally excluded. Qualified timing needs only the seven pinned
# wrapper files, not the potentially large diagnostic/artifact tree.
qualification_files = set()
for protocol_file in (args.protocol, args.evaluation_protocol):
    if not protocol_file:
        continue
    document = json.loads(pathlib.Path(protocol_file).read_bytes())
    for receipt in document.get("qualified_runtime", {}).get("evidence", {}).values():
        source = (ROOT / receipt["path"]).resolve()
        relative = source.relative_to(ROOT)
        if relative.parts[0] != "results":
            raise ValueError("qualification evidence must live under results/")
        if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["file_sha256"]:
            raise ValueError(f"qualification evidence changed: {relative}")
        qualification_files.add(relative)
if qualification_files:
    RUNTIME_EXCLUDES = [pattern for pattern in RUNTIME_EXCLUDES
                        if pattern not in ("results", "**/results/**", "results/hetero")]
    parents = {parent for path in qualification_files for parent in path.parents
               if parent != pathlib.Path(".")}
    for parent in sorted(parents, key=lambda path: (len(path.parts), str(path))):
        RUNTIME_EXCLUDES.extend((f"!{parent}", f"{parent}/**"))
    RUNTIME_EXCLUDES.extend(f"!{path}" for path in sorted(qualification_files))
SOURCES = sorted([
    *(ROOT / "hetero").rglob("*.py"),
    *(ROOT / "hetero").rglob("*.json"),
    ROOT / "scripts/hetero_ctl.py",
    ROOT / "deploy/k8s/omp-gateway.yaml",
    ROOT / "deploy/k8s/omp-router.yaml",
])


def source_digest() -> dict[str, str]:
    digests = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in SOURCES
        if path.exists()
    }
    digests.update({f"input:{key}": hashlib.sha256(raw.encode()).hexdigest()
                    for key, raw in input_env.items()})
    return digests

def _find_artifact_dirs(payload) -> list[str]:
    """Collects every remote artifact directory named anywhere in the payload."""
    found: list[str] = []
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            value = node.get("artifact_dir")
            if isinstance(value, str):
                found.append(value)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return sorted(set(found))


def fetch_artifacts(remote_dir: str) -> dict:
    """Streams a head-node artifact directory into the local results tree.

    Bulk per-iteration rows never travel through the structured stdout channel,
    so they are tarred out of the driver pod instead.
    """
    local_dir = ROOT / "results" / "hetero" / "artifacts" / pathlib.PurePosixPath(remote_dir).name
    local_dir.mkdir(parents=True, exist_ok=True)
    parent = str(pathlib.PurePosixPath(remote_dir).parent)
    leaf = pathlib.PurePosixPath(remote_dir).name
    command = [
        "kubectl", "-n", args.namespace, "exec", args.head_pod, "-c", "ray-head", "--",
        "tar", "czf", "-", "-C", parent, leaf,
    ]
    extract = subprocess.run(
        ["tar", "xzf", "-", "-C", str(local_dir.parent)],
        input=subprocess.run(command, check=True, capture_output=True).stdout,
        check=True,
        capture_output=True,
    )
    files = sorted(p.name for p in local_dir.glob("*"))
    print(
        f"[OMP_EVENT: HETERO_ARTIFACTS_FETCHED remote={remote_dir} local={local_dir} "
        f"files={len(files)}]",
        flush=True,
    )
    return {"remote": remote_dir, "local": str(local_dir), "files": files,
            "stderr": extract.stderr.decode()[-200:]}



@stage(name="gateway_resources", order_idx=0, expected_s=180.0)
def reconcile_gateway_resources(state, timing):
    """Apply only this deployment's manifests; preflight uses server dry-run."""
    import yaml

    kubectl = ["kubectl", "-n", args.namespace]
    head = json.loads(subprocess.run(
        [*kubectl, "get", "pod", args.head_pod, "-o", "json"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout)
    cluster = head["metadata"]["labels"]["ray.io/cluster"]
    service_account = head["spec"].get("serviceAccountName", "default")
    documents = []
    for filename in ("omp-gateway.yaml", "omp-router.yaml"):
        for document in yaml.safe_load_all((ROOT / "deploy/k8s" / filename).read_text()):
            document["metadata"]["namespace"] = args.namespace
            if document["kind"] == "Service" and document["metadata"]["name"] == "omp-gateway":
                document["spec"]["selector"]["ray.io/cluster"] = cluster
            if document["kind"] == "RoleBinding":
                for subject in document["subjects"]:
                    subject.update(name=service_account, namespace=args.namespace)
            if document["kind"] == "ConfigMap":
                envoy = yaml.safe_load(document["data"]["envoy.yaml"])
                for upstream in envoy["static_resources"]["clusters"]:
                    if upstream["name"] == "omp_gateway":
                        socket_address = upstream["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]["address"]["socket_address"]
                        socket_address["address"] = f"omp-gateway.{args.namespace}.svc.cluster.local"
                document["data"]["envoy.yaml"] = yaml.safe_dump(envoy, sort_keys=False)
            documents.append(document)
    envoy_config = next(d["data"]["envoy.yaml"] for d in documents
                        if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "omp-router-envoy")
    router = next(d for d in documents if d["kind"] == "Deployment"
                  and d["metadata"]["name"] == "omp-router")
    router["spec"]["template"]["metadata"].setdefault("annotations", {})[
        "omp.llm-d.ai/envoy-config-sha256"
    ] = hashlib.sha256(envoy_config.encode()).hexdigest()
    command = [*kubectl, "apply", "-f", "-"]
    if args.command == "preflight":
        command.append("--dry-run=server")
    result = subprocess.run(command, input=yaml.safe_dump_all(documents),
                            text=True, capture_output=True, check=True, timeout=120)
    timing.log(result.stdout.strip())
    if args.command == "deploy":
        subprocess.run([*kubectl, "rollout", "status", "deployment/omp-router",
                        "--timeout=120s"], check=True, timeout=150)


@stage(name="hetero_command", order_idx=1, expected_s=args.timeout)
def run_remote(state, timing):
    supervisor = state.get("supervisor")
    if supervisor is None:
        supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
        state.set("supervisor", supervisor)

    submission_id = state.get("submission_id")
    if submission_id is not None:
        previous = supervisor.get_job(submission_id)
        if previous is not None and previous.status in ("FAILED", "STOPPED", "SUCCEEDED"):
            submission_id = None
    if submission_id is None:
        state.set("submitted_source_digest", source_digest())
        submission_id = supervisor.submit_job(
            entrypoint=shlex.join(["python3", "-u", "-m", "hetero.cli", *remote_args]),
            runtime_env={
                "working_dir": str(ROOT),
                "excludes": RUNTIME_EXCLUDES,
                "pip": ["xds-protos==1.84.0"],
                "env_vars": {
                    "PYTHONUNBUFFERED": "1",
                    "OMP_HETERO_DATA_DIR": args.data_dir,
                    **input_env,
                },
            },
            metadata={
                "purpose": "heterogeneous-llmd-serving-fabric",
                "command": args.command,
                "sources": json.dumps(state.get("submitted_source_digest")),
            },
            name=f"hetero-{args.command}",
        )
        state.set("submission_id", submission_id)

    result = supervisor.supervise_until_completion(
        submission_id=submission_id,
        timeout_s=args.timeout,
        stage_name="hetero_command",
        state_store=state,
    )
    if not result["measurements"]:
        raise RuntimeError("remote command produced no structured measurement payload")

    payload = {
        "submission_id": submission_id,
        "status": result["status"],
        "remote_arguments": remote_args,
        "source_digest": state.get("submitted_source_digest"),
        "saved_at": time.time(),
        "measurements": result["measurements"],
    }
    artifacts = []
    for remote_dir in _find_artifact_dirs(result["measurements"]):
        try:
            artifacts.append(fetch_artifacts(remote_dir))
        except subprocess.CalledProcessError as exc:
            artifacts.append(
                {"remote": remote_dir, "error": (exc.stderr or b"").decode()[-300:]}
            )
    if artifacts:
        payload["artifacts"] = artifacts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(args.output)
    print(f"[OMP_EVENT: HETERO_RESULT_SAVED path={args.output}]", flush=True)
    state.set("submission_id", None)


if __name__ == "__main__":
    runner = StageRunner(silence_timeout_s=max(900.0, args.timeout))
    if args.command in ("deploy", "preflight"):
        runner.register(reconcile_gateway_resources)
    runner.register(run_remote)
    if not runner.run():
        raise SystemExit(1)
