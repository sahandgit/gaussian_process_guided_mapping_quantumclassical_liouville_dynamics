# Scientific Protocol

## Current software claim

This repository provides an installable and testable research pipeline for GP/RKHS density surrogates in mapping-basis quantum-classical dynamics.

The current scientific object is the pipeline itself, not a completed thesis or paper.

## Method under test

The default method is:

```text
PBME-MInt characteristic transport
+ full-density GP/RKHS surrogate
+ midpoint mapping-QCLE excess-density correction
```

The density-difference method is retained as an optional ablation.

## Required comparisons

For each selected incident momentum, compare:

- TDSE / split-operator reference;
- grid-QCLE reference;
- PBME-MInt trajectory baseline;
- GP-midpoint corrected dynamics.

## Required diagnostics

- diabatic populations;
- electronic coherence;
- nuclear means and variances;
- trace conservation;
- energy drift;
- GP fit quality;
- correction magnitude;
- cloud-vs-GP consistency.

## Recommended benchmark momenta

```text
P0 = 10, 40, 100
```

`P0=10` tests slow/strongly nonadiabatic behavior; `P0=100` tests fast transmission; `P0=40` provides a useful intermediate case.
