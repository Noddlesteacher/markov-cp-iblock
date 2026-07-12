from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import plot_simulation_results as plots
import run_simulation_analysis as analysis
import simulation_study as sim


class SimulationStudyWorkflowTests(unittest.TestCase):
    def test_validate_transition_matrix(self):
        sim.validate_transition_matrix(sim.P_SIM1, sim.PI_SIM1)

        bad_P = sim.P_SIM1.copy()
        bad_P[0, 0] = 0.90
        with self.assertRaisesRegex(ValueError, "row of P"):
            sim.validate_transition_matrix(bad_P, sim.PI_SIM1)

        bad_pi = np.array([0.5, 0.5], dtype=float)
        with self.assertRaisesRegex(ValueError, "pi length"):
            sim.validate_transition_matrix(sim.P_SIM1, bad_pi)

        bad_pi_sum = np.array([0.2, 0.2, 0.2], dtype=float)
        with self.assertRaisesRegex(ValueError, "pi must sum"):
            sim.validate_transition_matrix(sim.P_SIM1, bad_pi_sum)

    def test_simulate_markov_chain_uses_one_based_labels(self):
        rng = np.random.default_rng(123)
        sequence = sim.simulate_markov_chain(sim.P_SIM1, sim.PI_SIM1, 25, rng)

        self.assertEqual(len(sequence), 25)
        self.assertTrue(all(state in {1, 2, 3} for state in sequence))

    def test_one_full_sequence_gives_same_history_and_nested_futures(self):
        sequences = sim.generate_sequences(
            sim.P_SIM1,
            sim.PI_SIM1,
            n_sim=1,
            training_length=12,
            horizons=(1, 2, 3),
            seed=123,
        )
        slices = analysis.check_nested_history_and_future(
            sequences[0],
            training_length=12,
            horizons=(1, 2, 3),
        )

        self.assertEqual(slices[1][0], slices[2][0])
        self.assertEqual(slices[1][0], slices[3][0])
        self.assertEqual(slices[1][1], slices[2][1][:1])
        self.assertEqual(slices[2][1], slices[3][1][:2])

    def test_case2_uses_markov_simulator(self):
        generated = analysis.make_case2_sequence(
            training_length=12,
            horizons=(1, 2, 3),
            seed=123,
        )
        rng = np.random.default_rng(123)
        expected = sim.simulate_markov_chain(
            sim.P_SIM1,
            sim.PI_SIM1,
            length=15,
            rng=rng,
        )
        self.assertEqual(generated, expected)

    def test_sweden_diagnostic_sequence_is_all_state_one(self):
        sequence = analysis.deterministic_sequence("sweden", 10, (1, 2, 3))
        self.assertEqual(sequence, [1] * 13)

        slices = analysis.check_nested_history_and_future(sequence, 10, (1, 2, 3))
        for history, true_future in slices.values():
            self.assertTrue(all(state == 1 for state in history))
            self.assertTrue(all(state == 1 for state in true_future))

    def test_target_coverage_grid_includes_alpha_zero(self):
        self.assertIn(1.00, sim.DEFAULT_TARGET_COVERAGES)
        alphas = analysis.target_coverages_to_alphas(sim.DEFAULT_TARGET_COVERAGES)
        self.assertIn(0.0, alphas)

    def test_default_methods_exclude_legacy_factorial_method(self):
        sequence = [1, 2, 3, 1, 2, 3, 1, 2]
        rows = analysis.evaluate_history_for_horizons(
            full_sequence=sequence,
            training_length=5,
            horizons=(1,),
            target_coverages=(0.80,),
            max_permutations=5,
            randomized_ties=False,
            cp_seed=123,
            simulation_name="tiny",
            replicate=1,
        )
        self.assertEqual(
            {row["method"] for row in rows},
            {"original", "iblock_count"},
        )

    def test_summary_uses_raw_set_size_not_scaled_set_size(self):
        raw_df = pd.DataFrame(
            [
                {
                    "simulation": "fake",
                    "replicate": 1,
                    "horizon": 1,
                    "alpha": 0.2,
                    "target_coverage": 0.8,
                    "method": "original",
                    "covered": 1,
                    "set_size": 2,
                    "true_future": "(1)",
                    "cp_set": "(1) (2)",
                },
                {
                    "simulation": "fake",
                    "replicate": 2,
                    "horizon": 1,
                    "alpha": 0.2,
                    "target_coverage": 0.8,
                    "method": "original",
                    "covered": 0,
                    "set_size": 1,
                    "true_future": "(3)",
                    "cp_set": "(1)",
                },
            ]
        )

        summary_df = analysis.summarize_results(raw_df)
        self.assertNotIn("scaled_set_size", summary_df.columns)
        self.assertEqual(summary_df.loc[0, "mean_set_size"], 1.5)
        self.assertTrue((summary_df["empirical_coverage"] >= 0).all())
        self.assertTrue((summary_df["empirical_coverage"] <= 1).all())

    def test_plotting_works_from_summary_without_cp(self):
        summary_df = pd.DataFrame(
            [
                {
                    "simulation": "fake",
                    "method": method,
                    "horizon": horizon,
                    "alpha": 0.2,
                    "target_coverage": 0.8,
                    "n_sim": 2,
                    "empirical_coverage": 0.5,
                    "coverage_mcse": 0.25,
                    "mean_set_size": float(horizon),
                    "sd_set_size": 0.0,
                }
                for method in ("original", "iblock_count")
                for horizon in (1, 2, 3)
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            saved_paths = plots.create_outputs(summary_df, output_root)
            self.assertTrue(saved_paths)
            self.assertTrue(all(path.exists() for path in saved_paths))
            self.assertTrue((output_root / "figures" / "fake_reliability.png").exists())
            self.assertTrue(
                (output_root / "tables" / "fake_raw_set_size_summary.csv").exists()
            )

    def test_quick_pipeline_subprocess(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            subprocess.run(
                [
                    sys.executable,
                    "simulation_study.py",
                    "--quick",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            input_path = output_root / "input" / "sim1_sequences_quick.npz"
            self.assertTrue(input_path.exists())

            subprocess.run(
                [
                    sys.executable,
                    "run_simulation_analysis.py",
                    "--input",
                    str(input_path),
                    "--quick",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            summary_path = output_root / "output" / "sim1_summary_quick.csv"
            self.assertTrue(summary_path.exists())

            subprocess.run(
                [
                    sys.executable,
                    "plot_simulation_results.py",
                    "--summary",
                    str(summary_path),
                    "--output-dir",
                    tmpdir,
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertTrue((output_root / "figures" / "sim1_reliability.png").exists())
            self.assertTrue(
                (
                    output_root
                    / "tables"
                    / "sim1_raw_set_size_target_080.md"
                ).exists()
            )

    def test_diagnostic_cases_quick_subprocesses(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            for case_name in ("case2", "sweden"):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "run_simulation_analysis.py",
                        "--case",
                        case_name,
                        "--quick",
                        "--output-dir",
                        tmpdir,
                    ],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertIn("Saved diagnostic", completed.stdout)
                self.assertTrue(
                    (Path(tmpdir) / "output" / f"{case_name}_diagnostic.csv").exists()
                )

    def test_saved_cp_set_path_lengths_match_horizon(self):
        sequence = [1, 2, 3, 1, 2, 3, 1, 2]
        rows = analysis.evaluate_history_for_horizons(
            full_sequence=sequence,
            training_length=5,
            horizons=(1, 2),
            target_coverages=(0.80,),
            max_permutations=5,
            randomized_ties=False,
            cp_seed=123,
            simulation_name="tiny",
            replicate=1,
        )

        for row in rows:
            horizon = int(row["horizon"])
            for path_body in re.findall(r"\(([^)]*)\)", row["cp_set"]):
                path_length = 0 if path_body == "" else len(path_body.split(","))
                self.assertEqual(path_length, horizon)


if __name__ == "__main__":
    unittest.main()
