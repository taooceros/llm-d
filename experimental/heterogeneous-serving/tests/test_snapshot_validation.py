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


def _measured(kv='bfloat16'):
    return {'layout': 'S', 'manifest_sha256': 'x', 'prompts': [[1]], 'sampling': {'temperature': 0},
            'engine_identity': {'a': {'kv_cache_dtype': kv}}, 'outputs': {'a': [[1, 2]]}}


def test_greedy_reference_requires_engine_identity():
    ref = _measured()
    del ref['engine_identity']
    with pytest.raises(RuntimeError, match='no engine_identity'):
        validation._verify_greedy_reference(ref, _measured())


def test_greedy_reference_refuses_kv_dtype_mismatch():
    with pytest.raises(RuntimeError, match='engine_identity'):
        validation._verify_greedy_reference(_measured('float32'), _measured('bfloat16'))


def test_greedy_reference_accepts_equal():
    assert validation._verify_greedy_reference(_measured(), _measured())['passed'] is True


def test_agreement_refuses_different_prompts_and_records_drift_ungated():
    ref = _measured()
    other = {**_measured(), 'prompts': [[2]], 'cross_layout': {'passed': True}}
    with pytest.raises(RuntimeError, match='prompts'):
        validation._verify_agreement(ref, other, 'ref.json')
    divergences = [{'prompt_index': 3, 'd': 7, 'gap_nats': 0.25},
                   {'prompt_index': 9, 'd': 21, 'gap_nats': 9.1}]
    for passed in (True, False):
        cross = {'passed': passed, 'exact_prompts': 30, 'first_divergence_min': 7, 'divergences': divergences}
        res = validation._verify_agreement({**ref, 'layout': 'M'}, {**_measured(), 'cross_layout': cross}, 'ref.json')
        assert res['gated'] is False and 'passed' not in res
        assert res['tie_gate_passed'] is passed
        assert res['summary'] == {'exact_prompts': 30, 'first_divergence_min': 7, 'divergences': 2,
                                  'divergence_indices': [7, 21], 'tie_gaps_nats': [0.25, 9.1],
                                  'max_tie_gap_nats': 9.1}
        assert res['other_layout'] == 'M'


def test_full_bf16_reference_refuses_fp32_activation_reference():
    # engine_identity is the engine's model_build: the activation profile is part of it.
    full_bf16 = _measured()
    full_bf16['engine_identity'] = {'a': {'kv_cache_dtype': 'bfloat16', 'activation_dtype': None,
                                          'matmul_precision': 'default'}}
    fp32_act = _measured()
    fp32_act['engine_identity'] = {'a': {'kv_cache_dtype': 'bfloat16', 'activation_dtype': 'float32',
                                         'matmul_precision': 'highest'}}
    with pytest.raises(RuntimeError, match='engine_identity'):
        validation._verify_greedy_reference(fp32_act, full_bf16)
