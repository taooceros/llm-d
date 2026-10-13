import copy
import hashlib

import pytest

from scripts import snapshot_validate as validation


def fingerprints(payload=b'original snapshot payload'):
    shard = {'array': 0, 'shard': 0, 'path': "['weight']", 'device_id': 0,
             'index': [[None, None, None]], 'shape': [len(payload)],
             'dtype': 'uint8', 'bytes': len(payload),
             'sha256': hashlib.sha256(payload).hexdigest()}
    worker = {'process_rank': 0, 'bytes': len(payload), 'shards': [shard],
              'sha256': hashlib.sha256(payload).hexdigest()}
    cold = validation._host_measurements('file-fingerprint', 'S', [
        {'fingerprints': [{'instance': 'small_a', **worker}]}])
    live = {'layout': 'S', 'workers': {'small_a': [worker]}}
    return cold, live


def test_cold_host_fingerprints_are_valid_live_weight_references():
    cold, live = fingerprints()
    validation._verify_fingerprint_reference(cold, live)


@pytest.mark.parametrize('fault', ['layout', 'missing_layout', 'payload', 'missing_worker'])
def test_incomplete_or_mismatched_weight_references_are_refused(fault):
    cold, live = fingerprints()
    live = copy.deepcopy(live)
    if fault == 'layout':
        cold['layout'] = 'H'
    elif fault == 'missing_layout':
        del cold['layout']
    elif fault == 'payload':
        live['workers']['small_a'][0]['shards'][0]['sha256'] = hashlib.sha256(b'changed payload').hexdigest()
    else:
        live['workers'] = {}
    with pytest.raises(RuntimeError):
        validation._verify_fingerprint_reference(cold, live)
