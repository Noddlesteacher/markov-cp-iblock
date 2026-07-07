import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import plot_simulation_results as plots
import simulation_study as sim


class SimulationStudyTests(unittest.TestCase):
    def test_validate_transition_matrix(self):
        sim.validate_transition_matrix(sim.P_SIM1, sim.PI_SIM1)
        sim.validate_transition_matrix(sim.P_SIM4, sim.PI_SIM4)

        bad_P = np.array(
            [
                [0.8, 0.3, 0.0],
                [0.3, 0.6, 0.1],
                [0.2, 0.2, 0.6],
            ],
            dtype=float,
        )
        with self.assertRaisesRegex(ValueError, "row of P"):
            sim.validate_transition_matrix(bad_P, sim.PI_SIM1)

    def test_simulate_markov_chain_uses_one_based_labels(self):
        rng = np.random.default_rng(123)
        sequence = sim.simulate_markov_chain(sim.P_SIM1, sim.PI_SIM1, 25, rng)

        self.assertEqual(len(sequence), 25)
        self.assertTrue(all(state in {1, 2, 3} for state in sequence))

    def test_sim4_generates_only_state_one(self):
        rng = np.random.default_rng(123)
        sequence = sim.simulate_markov_chain(sim.P_SIM4, sim.PI_SIM4, 25, rng)

        self.assertEqual(sequence, [1] * 25)

    def test_quick_simulation_dataframe_and_path_lengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_df = sim.run_simulation_case(
                name="tiny",
                P=sim.P_SIM3,
                pi=sim.PI_SIM3,
                n_sim=1,
                T=12,
                horizons=[1, 2],
                alpha_values=[0.2],
                max_permutations=10,
                randomized_ties=False,
                master_seed=123,
                output_dir=Path(tmpdir),
            )

        self.assertEqual(
            set(raw_df["method"]),
            {"original", "permutation_count", "iblock_count"},
        )
        self.assertTrue((raw_df["set_size"] >= 0).all())

        for _, row in raw_df.iterrows():
            horizon = int(row["horizon"])
            for path_body in re.findall(r"\(([^)]*)\)", row["cp_set"]):
                path_length = 0 if path_body == "" else len(path_body.split(","))
                self.assertEqual(path_length, horizon)

    def test_summarize_results_outputs_valid_ranges(self):
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
                    "scaled_set_size": 2 / 3,
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
                    "scaled_set_size": 1 / 3,
                    "true_future": "(3)",
                    "cp_set": "(1)",
                },
            ]
        )

        summary_df = sim.summarize_results(raw_df)
        self.assertTrue((summary_df["empirical_coverage"] >= 0).all())
        self.assertTrue((summary_df["empirical_coverage"] <= 1).all())
        self.assertTrue((summary_df["mean_set_size"] >= 0).all())

    def test_plotting_script_runs_on_tiny_summary(self):
        raw_df = pd.DataFrame(
            [
                {
                    "simulation": "fake",
                    "replicate": 1,
                    "horizon": 1,
                    "alpha": 0.2,
                    "target_coverage": 0.8,
                    "method": method,
                    "covered": 1,
                    "set_size": 1,
                    "scaled_set_size": 1 / 3,
                    "true_future": "(1)",
                    "cp_set": "(1)",
                }
                for method in ("original", "permutation_count", "iblock_count")
            ]
        )
        summary_df = sim.summarize_results(raw_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            saved_paths = plots.plot_all(summary_df, output_dir)
            self.assertTrue(saved_paths)
            self.assertTrue(all(path.exists() for path in saved_paths))

    def test_quick_simulation_subprocess(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
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

            self.assertIn("Saved raw results", completed.stdout)
            self.assertTrue((Path(tmpdir) / "sim1_raw_results.csv").exists())
            self.assertTrue((Path(tmpdir) / "sim1_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
