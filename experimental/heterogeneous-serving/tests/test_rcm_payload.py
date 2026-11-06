"""The RCM supervisor scrapes one measurement payload per job log (hetero/evidence.py emits it).

Ray prints forwarded worker log lines from another thread. One such line once landed between the
payload and its MEASUREMENTS_EMITTED event (autoscale-098524d-pb-gcs-ov-r5). The deploy succeeded,
but the arm was failed because no payload was scraped.
"""
import json
import unittest

from hetero.evidence import BANNER
from omp_rcm.ray_job import RayJobSupervisor

RULE = "=" * 78
PAYLOAD = {"action": "deploy", "deployment": {"instances": [{"note": "a } b\nc"}]}, "ok": True}


def log(after_payload):
    body = json.dumps(PAYLOAD, separators=(",", ":"))
    return "\n".join(["(HeteroEngine pid=1, ip=10.0.0.1) INFO loading", "", RULE, BANNER, RULE, body, "",
                      *after_payload, '[OMP_EVENT: COMMAND_COMPLETE {"command":"deploy"}]', ""])


class PayloadScrapeTests(unittest.TestCase):
    def test_clean_log(self):
        text = log(['[OMP_EVENT: MEASUREMENTS_EMITTED {"bytes":1}]'])
        self.assertEqual(RayJobSupervisor.extract_json_measurements(text), PAYLOAD)

    def test_worker_line_interleaved_before_the_event(self):
        text = log(["\x1b[36m(pid=, ip=192.168.102.193)\x1b[0m learning/45eac/research/pjrt/tpu_pjrt_client.cc:1838"
                    '[OMP_EVENT: MEASUREMENTS_EMITTED {"bytes":1}]'])
        self.assertEqual(RayJobSupervisor.extract_json_measurements(text), PAYLOAD)

    def test_no_payload(self):
        self.assertEqual(RayJobSupervisor.extract_json_measurements("[INFO] nothing\n" + RULE + "\n"), {})


if __name__ == "__main__":
    unittest.main()
