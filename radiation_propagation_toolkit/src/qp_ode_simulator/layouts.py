"""Layout helpers for device-independent QP-ODE simulations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validate_coordinates(coords: np.ndarray) -> np.ndarray:
    """Return a validated ``(qubit, 2)`` coordinate array in millimetres."""
    coordinates = np.asarray(coords, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("coordinates must have shape (n_qubits, 2)")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must contain only finite values")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("coordinates must not contain duplicate qubit locations")
    return coordinates


def rectangular_layout(
    rows: int,
    columns: int,
    *,
    pitch_mm: float = 1.0,
    center_mm: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Create a centered rectangular qubit layout."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if not np.isfinite(pitch_mm) or pitch_mm <= 0:
        raise ValueError("pitch_mm must be finite and positive")
    yy, xx = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    coordinates = np.column_stack((xx.ravel(), yy.ravel())).astype(float) * pitch_mm
    coordinates -= coordinates.mean(axis=0)
    coordinates += np.asarray(center_mm, dtype=float)
    return validate_coordinates(coordinates)


def load_coordinates(path: str | Path) -> np.ndarray:
    """Load coordinates from ``.csv``, ``.json``, ``.npy``, or text files."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        coordinates = np.load(source)
    elif suffix == ".json":
        payload = json.loads(source.read_text())
        coordinates = payload.get("coordinates_mm", payload) if isinstance(payload, dict) else payload
    else:
        delimiter = "," if suffix == ".csv" else None
        try:
            coordinates = np.loadtxt(source, delimiter=delimiter)
        except ValueError:
            coordinates = np.loadtxt(source, delimiter=delimiter, skiprows=1)
    return validate_coordinates(coordinates)


# The plain simulator has a neutral, non-Sycamore default. Stim runs construct their
# coordinates directly from the requested QEC circuit.
DEFAULT_COORDS = rectangular_layout(5, 5, pitch_mm=1.0)
