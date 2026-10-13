import json
from pathlib import Path
import tempfile
import unittest

from scripts.fast_graph_summary import median_table, workers


class FastGraphSummaryTests(unittest.TestCase):
    def test_median_preserves_submillisecond_worker_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = {"measurements": {"deployment": {"instances": [{
                "descriptor": {"instance_id": "small_a"},
                "initialization": {"worker_init_phases": [
                    {"executor_rank": 0, "phases": {
                        "graph_build": {"start": 10.0, "end": 11.0004}}},
                    {"executor_rank": 1, "phases": {
                        "graph_build": {"start": 10.0, "end": 11.0008}}},
                ]},
            }]}}}
            (root / "S-cold-base-deploy.json").write_text(json.dumps(evidence))
            measured = median_table(workers(root, "S-cold-base"))
        self.assertAlmostEqual(measured["graph_build"], 1.0006, places=12)
