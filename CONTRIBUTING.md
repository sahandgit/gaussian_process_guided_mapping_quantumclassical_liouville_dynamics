# Contributing

This repository is a research pipeline. Contributions should preserve numerical reproducibility.

## Rules

1. Add tests for any change that alters dynamics, observables, GP fitting, serialization, or comparison outputs.
2. Do not change default scientific parameters silently.
3. Record changes to method assumptions in `docs/SCIENTIFIC_PROTOCOL.md`.
4. Keep generated results out of git unless they are intentionally curated examples.
5. Prefer small, reviewable commits.
