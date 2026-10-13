import json
import math

from scripts import probe_drift


def _artifact(tmp_path, name, layout, tables, engines=('a', 'b')):
    rows = [{'engine': e, 'prompt_index': i, 'prefix_len': 0,
             'top_logprobs': {str(t): lp for t, lp in table.items()}}
            for i, table in enumerate(tables) for e in engines]
    path = tmp_path / name
    path.write_text(json.dumps({'measurements': {'ok': True, 'action': 'probe', 'layout': layout,
                                                 'engine_identity': {}, 'rows': rows}}))
    return probe_drift.load(path)


def test_deviation_ignores_tokens_below_1e3_in_both_tables():
    a = {1: math.log(0.9), 2: math.log(0.0005), 3: math.log(0.0995)}
    b = {1: math.log(0.8), 2: math.log(0.0001), 3: math.log(0.1999)}
    assert probe_drift.deviation(a, b) == abs(math.log(0.0995) - math.log(0.1999))


def test_compare_reports_argmax_quantiles_and_tv(tmp_path):
    left = _artifact(tmp_path, 'l.json', 'S', [{1: -0.1, 2: -2.5}, {1: -0.7, 2: -0.7}])
    right = _artifact(tmp_path, 'r.json', 'H', [{1: -0.1, 2: -2.5}, {1: -1.0, 2: -0.45}])
    result = probe_drift.compare(left, right)
    assert result['argmax_equal'] == 1 and result['probes'] == 2
    assert math.isclose(result['max_abs_dlogprob']['max'], 0.3)
    assert result['max_abs_dlogprob']['p90'] == result['max_abs_dlogprob']['max']
    assert result['tv']['median'] > 0
    assert left['replicas_identical'] is True


def test_replica_mismatch_is_reported(tmp_path):
    path = tmp_path / 'x.json'
    rows = [{'engine': 'a', 'prompt_index': 0, 'prefix_len': 0, 'top_logprobs': {'1': -0.1}},
            {'engine': 'b', 'prompt_index': 0, 'prefix_len': 0, 'top_logprobs': {'1': -0.2}}]
    path.write_text(json.dumps({'measurements': {'ok': True, 'action': 'probe', 'layout': 'S',
                                                 'engine_identity': {}, 'rows': rows}}))
    assert probe_drift.load(path)['replicas_identical'] is False


def test_sign_test_matches_q1_record():
    # Q1: 21 vs 23 probes, p = 0.88.
    result = probe_drift.sign_test([1] * 21 + [0] * 23, [0] * 21 + [1] * 23)
    assert (result['first_larger'], result['second_larger']) == (21, 23)
    assert round(result['p_two_sided'], 2) == 0.88
