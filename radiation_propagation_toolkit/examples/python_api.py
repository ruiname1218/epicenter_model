"""Minimal custom-layout QP-ODE example."""

from qp_ode_simulator import load_default_simulator_config, rectangular_layout, run_simulation


config = load_default_simulator_config()
config["n_events"] = 2
config["seed"] = 11
coords = rectangular_layout(3, 4, pitch_mm=0.8)

result = run_simulation(config, coords_mm=coords)
result.save("example_output")

print("observations:", result.observations.shape)
print("QP density:", result.physics["x_qp"].shape)
print("T1:", result.physics["t1_us"].shape)
print("labels:", list(result.parameters.columns))
