import inspect
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

import numpy as np

import markov_cp_routines as m


DENSE_ADJACENCY = np.ones((3, 3), dtype=int)


class MarkovCPRoutineTests(unittest.TestCase):
    def test_adjacency_validation(self):
        np.testing.assert_array_equal(
            m.check_adjacency([[1, 0], [0, 1]]),
            np.array([[1, 0], [0, 1]]),
        )

        with self.assertRaises(ValueError):
            m.check_adjacency([[1, 0, 1], [0, 1, 0]])

        with self.assertRaises(ValueError):
            m.check_adjacency([[1, 2], [0, 1]])

        with self.assertRaises(ValueError):
            m.check_adjacency([[1, 0], [0, 0]])

    def test_candidate_enumeration_dense(self):
        self.assertEqual(
            m.enumerate_paths(1, 1, DENSE_ADJACENCY),
            [(1,), (2,), (3,)],
        )
        self.assertEqual(len(m.enumerate_paths(1, 2, DENSE_ADJACENCY)), 9)

    def test_i_block_decomposition_keeps_i0_and_tail_fixed(self):
        sequence = [2, 1, 3, 1, 2, 1, 3]
        initial_block, blocks, tail = m.split_i_blocks(sequence, anchor_state=1)

        self.assertEqual(initial_block, [2])
        self.assertEqual(blocks, [[1, 3], [1, 2]])
        self.assertEqual(tail, [1, 3])
        self.assertEqual(len(blocks), 2)

        permuted = m.build_sequence_from_blocks(
            initial_block,
            [blocks[1], blocks[0]],
            tail,
        )
        self.assertEqual(permuted, [2, 1, 2, 1, 3, 1, 3])

    def test_full_group_size_is_factorial_d(self):
        log_size, full_size = m.permutation_group_summary(4)
        self.assertEqual(full_size, math.factorial(4))
        self.assertAlmostEqual(log_size, math.log(math.factorial(4)))

    def test_zero_permutable_blocks_has_group_size_one(self):
        random.seed(11)
        row = m.iblock_candidate_diagnostic(
            history=[1, 1],
            candidate=(2,),
            adjacency=DENSE_ADJACENCY,
            max_permutations=None,
        )
        self.assertEqual(row.n_permutable_blocks, 0)
        self.assertEqual(row.full_group_size, 1)
        self.assertEqual(row.n_permutations_evaluated, 1)

    def test_block_orders_use_indices_not_resulting_sequences(self):
        blocks = [[1, 2], [1, 2]]
        orders = m.block_index_orders(2, max_permutations=None)
        self.assertEqual(orders, [(0, 1), (1, 0)])

        resulting_sequences = [
            m.build_sequence_from_blocks(None, [blocks[index] for index in order], [])
            for order in orders
        ]
        self.assertEqual(resulting_sequences[0], resulting_sequences[1])
        self.assertEqual(len(orders), 2)

    def test_dense_three_state_l1_table_sizes(self):
        history = [1, 2, 3, 1]

        random.seed(1)
        original_rows = m.original_iblock_table(
            history,
            horizon=1,
            adjacency=DENSE_ADJACENCY,
            max_permutations=20,
        )

        random.seed(1)
        auxiliary_rows = m.auxiliary_candidate_table(
            history,
            horizon=1,
            adjacency=DENSE_ADJACENCY,
            max_permutations=20,
        )

        self.assertEqual(len(original_rows), 3)
        self.assertEqual(len(auxiliary_rows), 9)
        self.assertEqual(
            [(row.original_candidate[0], row.auxiliary_state) for row in auxiliary_rows],
            [
                (1, 1), (1, 2), (1, 3),
                (2, 1), (2, 2), (2, 3),
                (3, 1), (3, 2), (3, 3),
            ],
        )

    def test_cardinality_weights_are_valid_for_each_y(self):
        history = [1, 2, 3, 1, 2, 1, 3]
        random.seed(2)
        rows = m.auxiliary_candidate_table(
            history,
            horizon=1,
            adjacency=DENSE_ADJACENCY,
            max_permutations=30,
        )

        totals: dict[tuple[int, ...], float] = {}
        for row in rows:
            self.assertTrue(math.isfinite(row.normalized_cardinality_weight))
            self.assertGreaterEqual(row.normalized_cardinality_weight, 0.0)
            totals[row.original_candidate] = (
                totals.get(row.original_candidate, 0.0)
                + row.normalized_cardinality_weight
            )

        for total in totals.values():
            self.assertAlmostEqual(total, 1.0)

    def test_p_values_and_q_tilde_are_probabilities(self):
        history = [1, 2, 3, 1, 2, 1, 3]

        random.seed(3)
        original_rows = m.original_iblock_table(
            history,
            horizon=1,
            adjacency=DENSE_ADJACENCY,
            max_permutations=30,
        )
        for row in original_rows:
            self.assertGreaterEqual(row.p_value, 0.0)
            self.assertLessEqual(row.p_value, 1.0)

        random.seed(3)
        results = m.cardinality_weighted_auxiliary_cp(
            history,
            horizon=1,
            alpha=0.2,
            adjacency=DENSE_ADJACENCY,
            max_permutations=30,
        )
        for result in results:
            self.assertGreaterEqual(result.q_tilde, 0.0)
            self.assertLessEqual(result.q_tilde, 1.0)
            self.assertEqual(result.included, result.q_tilde > 0.2)

    def test_same_external_seed_reproduces_results(self):
        history = [1, 2, 3, 1, 2, 1, 3]

        random.seed(123)
        first = m.cardinality_weighted_auxiliary_cp(
            history,
            horizon=1,
            alpha=0.2,
            adjacency=DENSE_ADJACENCY,
            max_permutations=30,
        )

        random.seed(123)
        second = m.cardinality_weighted_auxiliary_cp(
            history,
            horizon=1,
            alpha=0.2,
            adjacency=DENSE_ADJACENCY,
            max_permutations=30,
        )

        first_values = [
            (row.original_candidate, round(row.q_tilde, 12), row.included)
            for row in first
        ]
        second_values = [
            (row.original_candidate, round(row.q_tilde, 12), row.included)
            for row in second
        ]
        self.assertEqual(first_values, second_values)

    def test_same_external_seed_reproduces_auxiliary_rows(self):
        history = [1, 2, 3, 1, 2, 1, 3]

        random.seed(123)
        first = m.auxiliary_candidate_table(
            history,
            horizon=1,
            adjacency=DENSE_ADJACENCY,
            max_permutations=30,
        )

        random.seed(123)
        second = m.auxiliary_candidate_table(
            history,
            horizon=1,
            adjacency=DENSE_ADJACENCY,
            max_permutations=30,
        )

        first_values = [
            (
                row.original_candidate,
                row.auxiliary_state,
                row.extended_candidate,
                round(row.p_value, 12),
                row.n_permutable_blocks,
                round(row.log_full_group_size, 12),
                row.full_group_size,
                row.n_permutations_evaluated,
                round(row.normalized_cardinality_weight, 12),
            )
            for row in first
        ]
        second_values = [
            (
                row.original_candidate,
                row.auxiliary_state,
                row.extended_candidate,
                round(row.p_value, 12),
                row.n_permutable_blocks,
                round(row.log_full_group_size, 12),
                row.full_group_size,
                row.n_permutations_evaluated,
                round(row.normalized_cardinality_weight, 12),
            )
            for row in second
        ]
        self.assertEqual(first_values, second_values)

    def test_core_signatures_do_not_take_random_seed(self):
        functions = [
            m.iblock_candidate_diagnostic,
            m.original_iblock_table,
            m.original_iblock_prediction_set,
            m.auxiliary_candidate_table,
            m.cardinality_weighted_auxiliary_cp,
        ]

        for function in functions:
            self.assertNotIn("random_seed", inspect.signature(function).parameters)

    def test_run_demo_completes(self):
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "run_demo.py"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertIn("Editable history", completed.stdout)
        self.assertIn("Auxiliary extended-candidate table", completed.stdout)
        self.assertIn("Original versus cardinality-weighted auxiliary CP", completed.stdout)

    def test_quick_meeting_demo_completes(self):
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "quick_meeting_demo.py"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertIn("seed=", completed.stdout)
        self.assertIn("summary across seeds", completed.stdout)
        self.assertIn("q_tilde", completed.stdout)
        self.assertIn("(1,1)", completed.stdout)


if __name__ == "__main__":
    unittest.main()
