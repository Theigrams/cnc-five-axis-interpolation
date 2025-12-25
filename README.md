# cnc_five_axis_interpolation

[![Tests](https://github.com/Theigrams/cnc-five-axis-interpolation/actions/workflows/test.yml/badge.svg)](https://github.com/Theigrams/cnc-five-axis-interpolation/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-gh--pages-blue)](https://Theigrams.github.io/cnc-five-axis-interpolation/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

Python implementation of the C³-continuous five-axis trajectory generation algorithm from:

> Yuen, Zhang, Altintas (2013), *Smooth trajectory generation for five-axis machine tools*, International Journal of Machine Tools & Manufacture.

This project generates smooth five-axis toolpaths by **decoupling**:
- **Tool tip position**: Quintic B-spline + arc-length reparameterization (9th-order polynomial)
- **Tool orientation**: Quintic B-spline in spherical coordinates + 7th-order Bézier reparameterization

It also provides **A-C configuration inverse kinematics** mapping to machine coordinates `(X, Y, Z, A, C)`.

**[📖 Full Documentation](https://Theigrams.github.io/cnc-five-axis-interpolation/)**

---

## Algorithm Overview

### Inputs / Outputs

**Inputs**
- Discrete tool tip points `P_k` as `positions`: shape `(N, 3)` in mm
- Discrete tool axis unit vectors `O_k` as `orientations`: shape `(N, 3)`, normalized

**Outputs**
- A parametric path queried by **arc-length** `l ∈ [0, S]` (mm) returning:
  - Tool tip position `P(l)` (mm)
  - Tool axis orientation `O(l)` (unit vector)
  - Machine coordinates `(X(l), Y(l), Z(l), A(l), C(l))` for A-C configuration

### Algorithm Pipeline

1. **Position Spline (Section 2)**
   - Fit a quintic (degree-5) clamped B-spline through tool tip points using centripetal parameterization (Eq.4)
   - Compute arc-length numerically via adaptive Simpson integration (Eq.8-10)
   - Build piecewise 9th-degree feed-correction polynomial `u(l)` with C³ boundary conditions (Eq.11-22)
   - Adaptively subdivide if MSE exceeds tolerance (Eq.23)

2. **Orientation Spline (Section 3)**
   - Convert tool axis vectors to spherical angles `(θ, φ)` (Eq.25)
   - Fit quintic B-spline over `(θ, φ)` using angular parameterization (Eq.26-30)
   - Reparameterize by arc-length via 7th-degree Bézier `w(l)` with monotonicity + C³ constraints (Eq.32-41)
   - Map `(θ(w), φ(w))` back to Cartesian coordinates (Eq.31)

3. **Inverse Kinematics (Section 4)**
   - Convert `(P(l), O(l))` to machine axes `(X, Y, Z, A, C)` using closed-form A-C transformation (Eq.42)

---

## Installation

### Requirements
- Python **3.10+**
- `numpy`, `scipy`
- (optional) `pytest` for running tests

### Setup

```bash
# Clone and navigate to project
git clone https://github.com/Theigrams/cnc-five-axis-interpolation.git
cd cnc-five-axis-interpolation

# Install dependencies
python -m pip install numpy scipy

# (optional) Install test dependencies
python -m pip install pytest

# Add parent directory to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/.."
```

---

## Quick Start

```python
import numpy as np

from cnc_five_axis_interpolation import FiveAxisPath
from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

# Load sample data
positions, orientations, constraints = ijms2021_fan_shaped_path()

# Create and fit path
path = FiveAxisPath(
    positions=positions,
    orientations=orientations,
    mse_tolerance=1e-6,  # Feed-correction fit tolerance
    L_ac_z=70.0,         # A-to-C offset along Z (mm)
    L_Tya_z=150.0,       # Spindle-to-A offset along Z (mm)
)
path.fit()

# Sample uniformly along arc-length
l_values = np.linspace(0, path.length, 200)
XYZ, A, C = path.evaluate_machine_coords_batch(l_values)

print(f"Total arc length: {path.length:.2f} mm")
print(f"Machine coords shape: XYZ={XYZ.shape}, A={A.shape}, C={C.shape}")
```

**Notes:**
- `l` is arc-length in mm
- `A` and `C` are returned in radians
- Orientations are automatically normalized

---

## Module Structure

```
cnc_five_axis_interpolation/
├── __init__.py
├── algorithm.py                      # FiveAxisPath (main orchestrator)
├── core/
│   ├── position_spline.py            # Tool tip C³ spline + u(l) feed correction
│   ├── orientation_spline.py         # Tool axis C³ spline + w(l) Bézier reparam
│   └── kinematics.py                 # A-C inverse kinematics (Eq.42)
├── datasets/
│   ├── ijms2021.py                   # Fan-shaped discrete toolpath
│   └── jcde2022.py                   # Dual cubic B-spline toolpath
├── utils/
│   ├── integrals.py                  # Adaptive Simpson arc-length integration
│   └── geometry.py                   # Spherical/Cartesian conversions
└── tests/                            # Unit tests (pytest)
```

---

## API Reference

### `FiveAxisPath` (Main Entry Point)

```python
from cnc_five_axis_interpolation import FiveAxisPath

path = FiveAxisPath(positions, orientations, mse_tolerance=1e-6, L_ac_z=70.0, L_Tya_z=150.0)
path.fit()
```

**Methods:**
- `evaluate(l) -> (position, orientation)` - Single point evaluation
- `evaluate_batch(l_values) -> (positions, orientations)` - Batch evaluation
- `evaluate_machine_coords(l) -> (XYZ, A, C)` - Machine coordinates
- `evaluate_machine_coords_batch(l_values) -> (XYZ, A, C)` - Batch machine coords
- `sample_uniform(num_points) -> (l_values, positions, orientations)`

**Attributes:**
- `length` - Total arc length (mm)
- `position_spline` - PositionSpline object
- `orientation_spline` - OrientationSpline object

### `PositionSpline`

```python
from cnc_five_axis_interpolation.core.position_spline import PositionSpline

spline = PositionSpline(points, mse_tolerance=1e-6)
spline.fit()
position = spline.evaluate(l)  # Arc-length parameterized
```

### `OrientationSpline`

```python
from cnc_five_axis_interpolation.core.orientation_spline import OrientationSpline

spline = OrientationSpline(orientations, arc_lengths)
spline.fit()
orientation = spline.evaluate(l)  # Returns normalized unit vector
```

### `inverse_kinematics_ac`

```python
from cnc_five_axis_interpolation.core.kinematics import inverse_kinematics_ac

XYZ, A, C = inverse_kinematics_ac(P, O, L_ac_z=70.0, L_Tya_z=150.0)
```

---

## Paper Equation Cross-Reference

| Paper Equation | Topic | Code Location |
|----------------|-------|---------------|
| Eq.1-7 | Quintic B-spline interpolation | `core.bspline.fit_quintic_bspline` |
| Eq.4 | Centripetal parameterization | `core.bspline.centripetal_parameterization` |
| Eq.5 | Knot vector computation | `core.bspline.compute_knot_vector` |
| Eq.8-10 | Adaptive Simpson integration | `utils.integrals.adaptive_simpson` |
| Eq.11-22 | 9th-degree feed correction | `core.position_spline.fit_feed_correction_polynomial` |
| Eq.23 | Adaptive subdivision | `core.position_spline.PositionSpline._fit_feed_correction_adaptive` |
| Eq.25 | Cartesian → Spherical | `utils.geometry.cartesian_to_spherical` |
| Eq.26-30 | Orientation B-spline | `core.orientation_spline.fit_orientation_bspline` |
| Eq.28 | Angular parameterization | `core.bspline.angular_parameterization` |
| Eq.31 | Spherical → Cartesian | `utils.geometry.spherical_to_cartesian` |
| Eq.32-41 | Bézier reparameterization | `core.orientation_spline.BezierReparameterization` |
| Eq.42 | A-C inverse kinematics | `core.kinematics.inverse_kinematics_ac` |

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_position_spline.py -v
```

---

## Contributing

Issues and pull requests are welcome. Please run `python -m pytest -v` before submitting changes.

---

## Limitations

- Orientation reparameterization uses `scipy.optimize.minimize` (SLSQP) and falls back to initial feasible coefficients if optimization fails
- Jerk objective in Bézier reparameterization is a practical surrogate (finite-difference style)
- Inverse kinematics is specific to **A-C configuration** and its sign conventions

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Reference

Yuen, A., Zhang, K., & Altintas, Y. (2013). *Smooth trajectory generation for five-axis machine tools*. International Journal of Machine Tools and Manufacture, 71, 11-19.
