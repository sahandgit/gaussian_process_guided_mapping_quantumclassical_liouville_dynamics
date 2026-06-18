# Validation Report

This file records the current package-level validation status.

## Automated checks included

- Import tests for package modules.
- CLI contract tests for installed commands.
- Default-density-mode test confirming that full-density mode is the default.
- Tully derivative finite-difference check.
- Short-step PBME-MInt energy check.
- Midpoint effective-label semantics check.

## Current scope

These tests verify package integrity and basic numerical consistency. They do not replace a full convergence study.

## Required before scientific claims

- Time-step convergence.
- Training-cloud-size convergence.
- Seed variability.
- Comparison against TDSE and grid-QCLE for multiple `P0` values.
- Review of all approximation statements in future thesis or manuscript text.
