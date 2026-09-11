# Architecture

## 1. Event layer

Each event samples only event-level properties:

- epicenter and optional second source;
- circular or elliptical shape;
- stretched-exponential exponents and length scales;
- ballistic or diffusive apparent arrival law;
- source strength and lifetime.

The no-radiation hardware calibration is sampled once per device. Per-qubit fixed
`T1`, `T2`, and frequency variation remain constant across events.

## 2. Spatial and wavefront drive

The local radial profile is based on a stretched exponential,

```text
K(r,t) = exp(-(r/lambda(t))^beta(t)),
```

multiplied by a finite-width arrival front and an event source envelope. Elliptical
distance, spatially correlated residual fields, halo, boundary reflection, and two
sources are optional configuration branches. The default QEC profile uses one source
so that the epicenter label is unambiguous.

## 3. Local QP dynamics

For each qubit coordinate, the Riccati ODE is integrated with an exact constant-drive
step instead of forward Euler:

```text
dx/dt = g - s*x - r*x^2.
```

`s` is trapping, `r` is recombination, and `g` is the phenomenological local source.
The source lifetime is sampled independently from the QP trapping/recombination rates.

## 4. Hardware response

QP density changes relaxation through the transmon relation

```text
Delta_Gamma1 = (x_qp/pi) * sqrt(2*omega_01*Delta_Al/hbar).
```

The optional phase branch also emits QP-dependent pure-dephasing and frequency-shift
fields. It is disabled in the standard profile because it lacks direct calibration.

## 5. Observation and QEC

The continuous field may be sampled as Bernoulli qubit observations. For QEC, inverse
`T1/T2` rates are integrated/interpolated over physical circuit intervals. The
resulting Pauli channels are placed after each gate moment and before the next
measurement/TICK boundary. Stim generates ancilla records, detectors, and logical
observables from the actual circuit definition.

The decoder is disabled by default. When enabled, `fixed_circuit` builds one decoder
from a no-radiation reference and reuses it. Event-aware oracle decoding is deliberately
guarded because it is an optimistic upper bound rather than deployable performance.
