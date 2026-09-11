import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qp_ode_simulator import load_default_simulator_config, run_stim_pipeline
from qp_ode_simulator.configuration import load_json
from qp_ode_simulator.stim_qec import build_stim_layout, uses_fixed_hardware


STIM_AVAILABLE = importlib.util.find_spec("stim") is not None


def small_stim_config() -> dict:
    return {
        "schema_version": 1,
        "backend": "stim_surface_code",
        "task": "surface_code:rotated_memory_z",
        "distance": 3,
        "rounds": 3,
        "round_duration_ms": 0.001,
        "start_time_ms": 0.0,
        "window_role": "onset_and_peak",
        "shots_per_event": 1,
        "layout": {"qubit_pitch_mm": 1.0, "rotation_degrees": 0.0, "center_mm": [0, 0]},
        "circuit_noise": {
            "after_clifford_depolarization": 0.0,
            "before_round_data_depolarization": 0.0,
            "before_measure_flip_probability": 0.0,
            "after_reset_flip_probability": 0.0,
        },
        "gate_schedule": {
            "mode": "explicit",
            "tick_durations_ns": [30.0, 40.0, 40.0, 40.0, 40.0, 30.0, 780.0],
        },
        "noise_accounting": {"depolarizing_channels_exclude_t1_t2": True},
        "radiation_channel": {"targets": "all", "timing": "gate_slices"},
        "decoder": {"enabled": False, "mode": "fixed_circuit"},
        "seed": 7,
    }


@unittest.skipUnless(STIM_AVAILABLE, "Stim is optional")
class StimPipelineTests(unittest.TestCase):
    def test_layout_is_taken_from_circuit(self):
        layout = build_stim_layout(small_stim_config())
        self.assertEqual(len(layout.qubit_ids), 17)
        self.assertEqual(layout.physical_coords_mm.shape, (17, 2))

    def test_end_to_end_pipeline_and_save(self):
        config = load_default_simulator_config()
        config["n_events"] = 1
        config["time"] = {"start_ms": 0.0, "end_ms": 0.01, "dt_ms": 0.001}
        config["event_timing"] = {"onset_time_ms": 0.002, "control_fraction": 0.0}
        self.assertTrue(uses_fixed_hardware(config))
        result = run_stim_pipeline(config, small_stim_config())
        self.assertEqual(result.syndrome_arrays["detector_events"].shape[0:2], (1, 1))
        self.assertEqual(result.simulation.physics["t1_us"].shape, (1, 10, 17))
        with tempfile.TemporaryDirectory() as temporary:
            output = result.save(Path(temporary) / "stim")
            self.assertTrue((output / "stim_syndrome_events.npz").is_file())
            self.assertTrue((output / "stim_template_circuit.stim").is_file())
            self.assertTrue((output / "run_manifest.json").is_file())
            archive = np.load(output / "stim_syndrome_events.npz")
            self.assertIn("logical_observable_flips", archive.files)


if __name__ == "__main__":
    unittest.main()
