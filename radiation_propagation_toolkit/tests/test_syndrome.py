import unittest

import numpy as np

from qp_ode_simulator import rectangular_layout
from qp_ode_simulator.syndrome import (
    generate_syndromes,
    graph_repetition_checks,
    syndrome_from_faults,
    t1_to_pauli_probabilities,
)


class SyndromeTests(unittest.TestCase):
    def test_t1_t2_pauli_probabilities_are_normalized(self):
        t1 = np.full((1, 2, 3), 100.0)
        channel = t1_to_pauli_probabilities(t1, 0.001)
        total = channel["i"] + channel["x"] + channel["y"] + channel["z"]
        np.testing.assert_allclose(total, 1.0, atol=1e-7)
        np.testing.assert_allclose(channel["x"], channel["y"])

    def test_graph_checks_follow_geometry(self):
        coords = rectangular_layout(2, 2)
        checks, edges, check_coords = graph_repetition_checks(coords, 1.01)
        self.assertEqual(checks.shape, (4, 4))
        self.assertEqual(edges.shape, (4, 2))
        self.assertEqual(check_coords.shape, (4, 2))

    def test_single_fault_creates_detection_event(self):
        x_faults = np.zeros((1, 3, 2), dtype=bool)
        x_faults[0, 1, 0] = True
        result = syndrome_from_faults(
            x_faults,
            np.zeros_like(x_faults),
            np.zeros((0, 2), dtype=np.uint8),
            np.asarray([[1, 1]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result["z_detection_events"],
            np.asarray([[[0], [1], [0]]], dtype=np.uint8),
        )

    def test_t1_proxy_generation_returns_syndrome_arrays(self):
        coords = rectangular_layout(2, 2)
        probabilities = np.full((1, 4, 4), 0.1, dtype=np.float32)
        baseline = np.full_like(probabilities, 0.05)
        t1 = np.full_like(probabilities, 100.0)
        config = {
            "schema_version": 3,
            "backend": "graph_repetition_proxy",
            "graph_maximum_edge_mm": 1.01,
            "require_css_commutation": True,
            "data_error_model": "pauli_frame",
            "qec_round_duration_ms": 0.001,
            "fault_probability_backend": "t1_pauli",
            "measurement_error_probability": 0.0,
            "include_final_boundary": True,
            "output": {"include_intermediate_arrays": True},
            "seed": 1,
        }
        result, metadata = generate_syndromes(
            probabilities,
            baseline,
            coords,
            config,
            time_ms=np.arange(4) * 0.001,
            t1_us=t1,
        )
        self.assertEqual(metadata["fault_probability_backend"], "t1_pauli")
        self.assertIn("z_detection_events", result)
        self.assertIn("pauli_x_probability", result)


if __name__ == "__main__":
    unittest.main()
