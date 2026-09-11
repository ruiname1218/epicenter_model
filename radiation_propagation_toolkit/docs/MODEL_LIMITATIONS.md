# Model limitations and valid claims

## Appropriate uses

- Generate labeled, correlated superconducting-QEC stress-test datasets.
- Sweep event shape, apparent speed, range, strength, and epicenter.
- Train and benchmark epicenter or event-property estimators on known simulation truth.
- Test QEC sensitivity, detector patterns, quarantine strategies, and decoder robustness.
- Perform controlled ablations of trapping, recombination, source lifetime, and layout.

## Claims the current model does not support

- Absolute cosmic-ray or gamma event rates.
- Deposited energy, dose, or energy conservation across materials.
- Microscopic phonon or quasiparticle transport inside a real chip stack.
- True phonon group velocity from the fitted apparent response speed.
- Exact logical-error rates for a target processor without measured circuit calibration.
- Generalization to neutral atoms, ions, or photonics.

## Main approximation boundaries

1. `g(t,q)` is a phenomenological source proxy, not a Geant4/G4CMP transport output.
2. Local QP ODEs do not exchange QPs between qubit electrodes.
3. Several ODE/source parameters are not uniquely identifiable from the available
   response traces; source lifetime and recovery rates must not be overinterpreted.
4. Stim requires Pauli noise. Pauli twirling discards the directionality of amplitude
   damping and approximates coherent frequency shifts.
5. The standard profile is a generic prior guided by superconducting-device literature,
   not an absolute calibration to one production QPU.

## Before a quantitative hardware study

Replace the reference configuration with measured qubit `T1/T2/frequency`, gate and
readout durations, residual circuit noise, and detector data. Freeze calibration before
evaluating held-out events. Report both simulation-truth performance and real-data
performance; simulation-only localization accuracy is not evidence of real localization.
