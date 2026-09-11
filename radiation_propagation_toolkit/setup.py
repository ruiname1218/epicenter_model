"""Compatibility shim for editable installs with older pip/setuptools."""

from setuptools import find_packages, setup


setup(
    name="qp-ode-simulator",
    version="0.1.0",
    description="Phenomenological QP-ODE radiation and superconducting-QEC toolkit",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"qp_ode_simulator": ["configs/*.json"]},
    python_requires=">=3.10",
    install_requires=["numpy>=1.23,<3", "pandas>=1.5,<3", "scipy>=1.9,<2"],
    extras_require={
        "qec": ["stim==1.16.0", "pymatching==2.4.0"],
        "plot": ["matplotlib>=3.6,<4"],
        "dev": ["pytest>=8,<9"],
        "localization": ["scikit-learn>=1.4,<2"],
        "temporal": ["scikit-learn>=1.4,<2", "torch>=2.6,<3"],
        "all": [
            "stim==1.16.0",
            "pymatching==2.4.0",
            "matplotlib>=3.6,<4",
            "pytest>=8,<9",
            "scikit-learn>=1.4,<2",
        ],
    },
    entry_points={
        "console_scripts": [
            "qp-ode-simulate=qp_ode_simulator.cli.simulate:main",
            "qp-ode-stim=qp_ode_simulator.cli.stim:main",
            "qp-ode-syndrome=qp_ode_simulator.cli.syndrome:main",
            "qp-ode-validate-channel=qp_ode_simulator.cli.validate_channel:main",
            "qp-ode-localize=qp_ode_simulator.cli.localize:main",
            "qp-ode-dataset=qp_ode_simulator.cli.dataset:main",
        ]
    },
)
