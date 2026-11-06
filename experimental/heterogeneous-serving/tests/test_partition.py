"""Partition mode: a campaign that owns only some hosts of the slice (switch_campaign.sh AS_PART).

Everything it does must stay on its hosts and instances. That covers host actions, preflight
verdicts, registry actions, deploys and gateway handling. Another campaign's engines, placement
groups and state on the other hosts must be invisible to it.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

from hetero import cli, deployment
from scripts import snapshot_validate as validation

OURS = ["10.0.0.1", "10.0.0.2"]
THEIRS = ["10.0.0.3", "10.0.0.4"]


class FakeStrategy:
    def __init__(self, node_id, soft):
        self.node_id = node_id


class FakeCluster:
    """Just enough of ray for snapshot_validate.remote: 4 TPU hosts, 2 of them ours."""

    def __init__(self, pgs=(), registered=()):
        self.calls = []
        self.pgs = {str(i): pg for i, pg in enumerate(pgs)}
        self.registered = list(registered)

    def module(self):
        ray = types.ModuleType("ray")
        ray.init = lambda **kw: None
        ray.nodes = lambda: [{"Alive": True, "Resources": {"TPU": 4.0}, "NodeID": f"n-{ip}",
                              "NodeManagerAddress": ip} for ip in OURS + THEIRS]
        ray.get = lambda value, timeout=None: value

        def remote(**kwargs):
            def wrap(fn):
                def options(scheduling_strategy):
                    def call(*args):
                        self.calls.append(scheduling_strategy.node_id)
                        return fn(*args)
                    return SimpleNamespace(remote=call)
                return SimpleNamespace(options=options)
            return wrap
        ray.remote = remote
        util = types.ModuleType("ray.util")
        util.placement_group_table = lambda: self.pgs
        strategies = types.ModuleType("ray.util.scheduling_strategies")
        strategies.NodeAffinitySchedulingStrategy = FakeStrategy
        ray.util = util
        return {"ray": ray, "ray.util": util, "ray.util.scheduling_strategies": strategies}

    def registry(self):
        return SimpleNamespace(snapshot=SimpleNamespace(
            remote=lambda layout=None: [r for r in self.registered if layout in (None, r["layout_id"])]))


def run_remote(cluster, *argv):
    parser_args = ["snapshot_validate.py", *argv, "--remote"]
    emitted = []
    with mock.patch.dict(sys.modules, cluster.module()), \
            mock.patch("hetero.registry.find_registry", return_value=cluster.registry()), \
            mock.patch("hetero.evidence.emit_measurements", side_effect=emitted.append), \
            mock.patch.object(validation, "host_action",
                              side_effect=lambda action, *rest: {"action": action}), \
            mock.patch.object(sys, "argv", parser_args):
        validation.main()
    return emitted[0]


def pg(name, *ips, state="CREATED"):
    return {"name": name, "state": state, "bundles_to_node_id": {i: f"n-{ip}" for i, ip in enumerate(ips)}}


class SnapshotValidatePartitionTests(unittest.TestCase):
    def test_host_actions_touch_only_owned_hosts(self):
        cluster = FakeCluster()
        result = run_remote(cluster, "evict-pagecache", "--layout", "H", "--hosts", ",".join(OURS))
        self.assertEqual(sorted(cluster.calls), [f"n-{ip}" for ip in OURS])
        self.assertEqual(sorted(r["node_ip"] for r in result["hosts"]), OURS)
        self.assertEqual(result["partition_hosts"], OURS)
        whole = FakeCluster()
        run_remote(whole, "evict-pagecache", "--layout", "H")
        self.assertEqual(len(whole.calls), 4)

    def test_preflight_ignores_the_other_partition_but_not_ours(self):
        theirs = FakeCluster(pgs=[pg("omp-hetero--H--large_a", *THEIRS)],
                             registered=[{"instance_id": "large_a", "layout_id": "H"}])
        result = run_remote(theirs, "preflight", "--layout", "S", "--hosts", ",".join(OURS),
                            "--instances", "large_b")
        self.assertTrue(result["ok"])
        for cluster, error in (
                (FakeCluster(pgs=[pg("omp-hetero--H--large_b", *OURS)]), "placement groups: omp-hetero--H--large_b"),
                (FakeCluster(pgs=[pg("pending", state="PENDING")]), "placement groups: pending"),
                (FakeCluster(registered=[{"instance_id": "large_b", "layout_id": "H"}]), "registered engines: large_b")):
            with self.assertRaisesRegex(RuntimeError, error):
                run_remote(cluster, "preflight", "--layout", "S", "--hosts", ",".join(OURS), "--instances", "large_b")
        with self.assertRaisesRegex(RuntimeError, "registered engines: large_a"):  # whole slice: anything live refuses
            run_remote(theirs, "preflight", "--layout", "S")
        with self.assertRaisesRegex(RuntimeError, "placement groups: omp-hetero--H--large_a"):
            run_remote(FakeCluster(pgs=[pg("omp-hetero--H--large_a", *THEIRS)]), "preflight", "--layout", "S")

    def test_registry_actions_see_only_the_owned_instance(self):
        cluster = FakeCluster(registered=[{"instance_id": "large_a", "layout_id": "H"}])
        with self.assertRaisesRegex(RuntimeError, "incomplete H registry"):
            run_remote(cluster, "check", "--layout", "H", "--hosts", ",".join(OURS), "--instances", "large_b")

    def test_partition_refuses_unscoped_cleanup_and_unknown_hosts(self):
        with self.assertRaisesRegex(RuntimeError, "needs --run-scope"):
            run_remote(FakeCluster(), "cleanup", "--layout", "H", "--hosts", ",".join(OURS))
        with self.assertRaisesRegex(RuntimeError, "not live TPU hosts"):
            run_remote(FakeCluster(), "memory", "--layout", "H", "--hosts", "10.9.9.9")
        with self.assertRaises(SystemExit):
            run_remote(FakeCluster(), "check", "--layout", "H", "--hosts", ",".join(OURS))


class DeployPartitionTests(unittest.TestCase):
    def _check(self, registered, gateway):
        registry = SimpleNamespace(get_layout=SimpleNamespace(remote=lambda layout_id: registered))
        with mock.patch.object(deployment, "ray", SimpleNamespace(get=lambda value, timeout=None: value)), \
                mock.patch("hetero.gateway.lifecycle.find_gateway", return_value=gateway):
            deployment._check_partition(registry, "H", "digest-1")

    def test_partition_check(self):
        self._check(None, None)
        self._check({"layout_digest": "digest-1"}, None)  # the other partition registered H unchanged
        with self.assertRaisesRegex(RuntimeError, "cannot change the registered static layout"):
            self._check({"layout_digest": "digest-0"}, None)
        with self.assertRaisesRegex(RuntimeError, "live gateway"):
            self._check(None, object())

    def test_deploy_needs_instances_for_partition(self):
        with self.assertRaisesRegex(ValueError, "names its instances"):
            deployment.deploy(None, None, partition=True)

    def test_cli_deploy_flags_come_together(self):
        for instances, partition, error in ((None, True, "name --instances"), ("large_b", False, "add --partition")):
            args = argparse.Namespace(instances=instances, partition=partition, state_delivery=None)
            with mock.patch.object(cli, "_resolve_inventory", return_value=(None, "x")), \
                    mock.patch.object(cli, "_deployment_config", return_value=None):
                with self.assertRaisesRegex(ValueError, error):
                    cli.cmd_deploy(args)

    def test_cli_partition_deploy_skips_the_gateway(self):
        args = argparse.Namespace(instances="large_b", partition=True, state_delivery=None)
        with mock.patch.object(cli, "_resolve_inventory", return_value=(None, "x")), \
                mock.patch.object(cli, "_deployment_config", return_value="config"), \
                mock.patch.object(cli, "deploy", return_value={"ok": 1}) as deploy, \
                mock.patch.object(cli, "cmd_gateway_up") as gateway_up:
            report = cli.cmd_deploy(args)
        deploy.assert_called_once_with(None, "config", state_delivery=None, instance_ids=["large_b"], partition=True)
        gateway_up.assert_not_called()
        self.assertIsNone(report["gateway"])


def test_rcm_dir_follows_the_environment():
    code = "from omp_rcm import core, inspect; print(core.BASE_RCM_DIR, inspect.BASE_RCM_DIR)"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "OMP_RCM_DIR": "/tmp/omp_rcm-test-partition"}
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["/tmp/omp_rcm-test-partition"] * 2
    env.pop("OMP_RCM_DIR")
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["/tmp/omp_rcm"] * 2


@pytest.mark.skipif(not os.path.exists("/proc/locks"), reason="needs Linux flock")
def test_partition_leases_exclude_overlapping_hosts_and_whole_slice_runs():
    """The lock protocol itself: shared cluster lock + exclusive host locks, held by a flock chain."""
    with tempfile.TemporaryDirectory() as tmp:
        lock, host = os.path.join(tmp, "cluster.lock"), os.path.join(tmp, "10.0.0.1.lock")
        ready = os.path.join(tmp, "ready")
        holder = subprocess.Popen(["flock", "-o", "-s", lock, "flock", "-o", host, "sh", "-c",
                                   ': > "$1"; sleep 30', "hetero-lease", ready])
        try:
            for _ in range(100):
                if os.path.exists(ready):
                    break
                subprocess.run(["sleep", "0.05"])
            probe = lambda *a: subprocess.run(["flock", "-n", *a, "true"]).returncode  # noqa: E731
            assert probe("-s", lock) == 0        # another partition campaign gets the cluster lock...
            assert probe(host) != 0              # ...but not an owned host
            assert probe(lock) != 0              # a whole-slice campaign waits for every partition
            assert probe(os.path.join(tmp, "10.0.0.3.lock")) == 0
        finally:
            subprocess.run(["pkill", "-f", f"hetero-lease {ready}"])
            holder.wait(timeout=10)
        assert subprocess.run(["flock", "-n", lock, "true"]).returncode == 0  # released with the chain
