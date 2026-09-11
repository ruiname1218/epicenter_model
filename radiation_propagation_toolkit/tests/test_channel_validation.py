import unittest

import numpy as np

from qp_ode_simulator.channel_validation import (
    exact_gad_dephasing_kraus,
    ptgad_kraus,
)


class ChannelValidationTests(unittest.TestCase):
    @staticmethod
    def apply(state, channel):
        return sum(
            (operator @ state @ operator.conj().T for operator in channel),
            np.zeros_like(state),
        )

    def test_kraus_sets_preserve_trace(self):
        state = np.asarray([[0.3, 0.1], [0.1, 0.7]], dtype=np.complex128)
        for channel in (
            exact_gad_dephasing_kraus(100.0, 150.0, 1.0),
            ptgad_kraus(100.0, 150.0, 1.0),
        ):
            output = self.apply(state, channel)
            self.assertAlmostEqual(float(np.trace(output).real), 1.0)

    def test_exact_zero_temperature_channel_does_not_excite_ground_state(self):
        ground = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        output = self.apply(ground, exact_gad_dephasing_kraus(10.0, 20.0, 1.0))
        self.assertAlmostEqual(float(output[1, 1].real), 0.0)


if __name__ == "__main__":
    unittest.main()
