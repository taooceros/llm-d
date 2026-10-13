import unittest

from hetero import agreement as ag
from hetero.engine_actor import EngineSettings

C = ag.CROSS_LAYOUT_CRITERIA


def table(gap_other=0.3):
    return {1: -0.1, 2: -0.1 - gap_other, 3: -5.0}


class AgreementTests(unittest.TestCase):
    def test_first_divergence(self):
        self.assertIsNone(ag.first_divergence([1, 2], [1, 2]))
        self.assertEqual(ag.first_divergence([0] * 6, [0] * 5 + [1]), 5)
        self.assertEqual(ag.first_divergence([1, 2, 3], [1, 2]), 2)

    def test_replica_equality(self):
        out = {'a': [[1]], 'b': [[1]], 'c': [[1]], 'd': [[2]]}
        tp = {'a': 8, 'b': 8, 'c': 16, 'd': 8}
        r = ag.replica_equality(out, tp)
        self.assertFalse(r['passed'])
        self.assertFalse(r['groups']['8']['passed'])
        self.assertTrue(r['groups']['16']['passed'])
        self.assertEqual(r['groups']['8']['engines'], ['a', 'b', 'd'])
        out['d'] = [[1]]
        self.assertTrue(ag.replica_equality(out, tp)['passed'])
        self.assertEqual(ag.canonical_engine({'b': 1, 'a': 2}), 'a')

    def test_divergence_points(self):
        p = ag.divergence_points([[1, 2, 3], [4]], [[1, 9, 3], [4, 5]])
        self.assertEqual(p[0]['token_self'], 2)
        self.assertEqual(p[0]['token_other'], 9)
        self.assertEqual((p[1]['d'], p[1]['token_self'], p[1]['token_other']), (1, None, 5))
        with self.assertRaises(ValueError):
            ag.divergence_points([[1]], [])

    def test_tie_check(self):
        self.assertTrue(ag.tie_check(table(0.3), 1, 2, C)['passed'])
        self.assertAlmostEqual(ag.tie_check(table(0.3), 1, 2, C)['gap_nats'], 0.3)
        self.assertFalse(ag.tie_check(table(3.0), 1, 2, C)['passed'])
        self.assertFalse(ag.tie_check(table(), 1, 99, C)['passed'])
        self.assertTrue(ag.tie_check({'1': -0.1, '2': -0.4}, 1, 2, C)['passed'])
        self.assertFalse(ag.tie_check({1: -0.1, 2: float('nan')}, 1, 2, C)['passed'])
        self.assertFalse(ag.tie_check({1: -0.1, 2: -float('inf')}, 1, 2, C)['passed'])
        self.assertTrue(ag.tie_check({7: -0.1, 1: -0.5, 2: -0.6}, 1, 2, C)['passed'])
        self.assertTrue(ag.tie_check(table(), None, None, C)['passed'])
        self.assertFalse(ag.tie_check(table(), 1, None, C)['passed'])

    def diag(self, points, late_gap=0.3):
        return [{'prompt_index': i, 'd': d, 'token_id': 1, 'top_logprobs': table(late_gap)}
                for i, d in points]

    def side(self, n_div, d=10, n=32):
        a = [[1] * 12 for _ in range(n)]
        b = [list(x) for x in a]
        for i in range(n_div):
            b[i][d] = 2
        a2 = [list(x) for x in a]
        for i in range(n_div):
            a2[i][d] = 1
        return a, b

    def test_evaluate_side(self):
        a, _ = self.side(0)
        r = ag.evaluate_side(a, a, [])
        self.assertTrue(r['passed'])
        self.assertEqual(r['divergences'], [])
        self.assertIsNone(r['first_divergence_min'])
        a, b = self.side(15)
        r = ag.evaluate_side(a, b, self.diag([(i, 10) for i in range(15)]))
        self.assertTrue(r['passed'])
        self.assertEqual(r['exact_prompts'], 17)
        a, b = self.side(17)
        r = ag.evaluate_side(a, b, self.diag([(i, 10) for i in range(17)]))
        self.assertFalse(r['checks']['exact_prompts'])
        self.assertFalse(r['passed'])
        a, b = self.side(3, d=2)
        r = ag.evaluate_side(a, b, self.diag([(i, 2) for i in range(3)]))
        self.assertFalse(r['checks']['early_divergences'])
        self.assertFalse(r['passed'])
        a, b = self.side(2)
        r = ag.evaluate_side(a, b, self.diag([(0, 10)]))
        self.assertFalse(r['passed'])
        self.assertIn('error', r)


class CompareDistributionsTests(unittest.TestCase):
    def test_identical(self):
        r = ag.compare_distributions(table(), table())
        self.assertTrue(r['argmax_equal'])
        self.assertEqual(r['shared'], 3)
        self.assertEqual(r['max_abs_dlogprob_shared'], 0)
        self.assertEqual(r['top1_a_lp_in_b'], -0.1)

    def test_disjoint(self):
        r = ag.compare_distributions({1: -0.1, 2: -2.0}, {3: -0.5})
        self.assertEqual(r['shared'], 0)
        self.assertIsNone(r['max_abs_dlogprob_shared'])
        self.assertIsNone(r['top1_a_lp_in_b'])
        self.assertFalse(r['argmax_equal'])

    def test_str_keys(self):
        r = ag.compare_distributions({'1': -0.1, '2': -1.0}, {1: -0.3, 2: -0.9})
        self.assertEqual((r['argmax_a'], r['argmax_b']), (1, 1))
        self.assertAlmostEqual(r['max_abs_dlogprob_shared'], 0.2)
        self.assertEqual(r['top1_a_lp_in_b'], -0.3)


class EngineSettingsTests(unittest.TestCase):
    def test_kv_dtype(self):
        self.assertEqual(EngineSettings().kv_cache_dtype, 'bfloat16')
        self.assertEqual(EngineSettings(kv_cache_dtype='float32').kv_cache_dtype, 'float32')
        for bad in ('auto', 'fp8'):
            with self.assertRaises(ValueError):
                EngineSettings(kv_cache_dtype=bad)


if __name__ == '__main__':
    unittest.main()
