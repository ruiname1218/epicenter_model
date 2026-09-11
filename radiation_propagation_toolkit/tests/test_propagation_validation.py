import copy

import numpy as np
import pandas as pd

from qp_ode_simulator.configuration import config_path, load_json
from qp_ode_simulator.propagation_validation import run_validation


def test_propagation_validation_is_seeded_and_returns_finite_recoveries():
    config = copy.deepcopy(load_json(config_path("propagation_validation")))
    config["trials_per_law"] = 3

    first_frame, first_summary = run_validation(config)
    second_frame, second_summary = run_validation(config)

    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert first_summary == second_summary
    assert len(first_frame) == 6
    assert set(first_frame["law"]) == {"ballistic", "diffusive"}
    columns = [
        "propagation_value_estimated",
        "lambda_estimated_mm",
        "epicenter_error_mm",
        "arrival_rmse_us",
    ]
    assert np.isfinite(first_frame[columns].to_numpy()).all()
    assert first_frame["fit_success"].all()
    assert first_summary["field_level"] is True
    assert first_summary["binary_shot_noise_included"] is False
