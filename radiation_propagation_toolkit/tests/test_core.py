import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qp_ode_simulator import (
    config_path,
    load_default_simulator_config,
    rectangular_layout,
    run_simulation,
    solve_qp_dynamics,
)


def small_config() -> dict:
    config = load_default_simulator_config()
    config["n_events"] = 3
    config["seed"] = 123
    config["time"] = {"start_ms": 0.0, "end_ms": 0.02, "dt_ms": 0.001}
    config["event_timing"] = {"onset_time_ms": [0.002, 0.004], "control_fraction": 1 / 3}
    return config


class CoreTests(unittest.TestCase):
    def test_bundled_config_resolves_profile_inheritance(self):
        config = load_default_simulator_config()
        self.assertEqual(config["temporal_model"]["backend"], "qp_ode")
        self.assertEqual(config["hardware_hierarchy"]["mode"], "fixed_device")
        self.assertTrue(config_path("qp_ode_generic").is_file())

    def test_exact_ode_step_matches_pure_trapping_solution(self):
        time_ms = np.asarray([0.0, 0.1, 0.2])
        generation = np.full((3, 2), 1e-6)
        density = solve_qp_dynamics(time_ms, generation, 0.05, 0.0)
        equilibrium = 1e-6 / 0.05
        expected = equilibrium * (1 - np.exp(-0.05 * 100.0))
        np.testing.assert_allclose(density[1], expected, rtol=1e-12)

    def test_arbitrary_layout_result_is_reproducible_and_bounded(self):
        config = small_config()
        coords = rectangular_layout(2, 3, pitch_mm=0.8)
        first = run_simulation(config, coords_mm=coords)
        second = run_simulation(copy.deepcopy(config), coords_mm=coords)
        np.testing.assert_array_equal(first.observations, second.observations)
        np.testing.assert_allclose(first.physics["x_qp"], second.physics["x_qp"])
        self.assertEqual(first.observations.shape, (3, 20, 6))
        self.assertTrue(np.all((first.probabilities >= 0) & (first.probabilities <= 1)))
        self.assertTrue(np.all(first.physics["t1_us"] > 0))

    def test_fixed_device_calibration_is_shared_across_events(self):
        result = run_simulation(small_config(), coords_mm=rectangular_layout(2, 2))
        t1_maps = result.parameters["qp_baseline_t1_by_qubit_us"].tolist()
        t2_maps = result.parameters["qp_baseline_t2_by_qubit_us"].tolist()
        frequency_maps = result.parameters["qp_qubit_frequency_by_qubit_ghz"].tolist()
        self.assertEqual(len(set(t1_maps)), 1)
        self.assertEqual(len(set(t2_maps)), 1)
        self.assertEqual(len(set(frequency_maps)), 1)

    def test_result_saves_portable_files(self):
        result = run_simulation(small_config(), coords_mm=rectangular_layout(2, 2))
        with tempfile.TemporaryDirectory() as temporary:
            output = result.save(Path(temporary) / "dataset")
            self.assertTrue((output / "simulated_events.npz").is_file())
            self.assertTrue((output / "true_parameters.csv").is_file())
            self.assertTrue((output / "resolved_config.json").is_file())
            archive = np.load(output / "simulated_events.npz")
            self.assertIn("x_qp", archive.files)
            self.assertIn("t1_us", archive.files)


if __name__ == "__main__":
    unittest.main()
