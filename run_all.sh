#!/usr/bin/env bash
# Reproduce every number and figure in the README (about 20-30 min on one CPU core).
set -e
python -m pytest -q
python -m experiments.gradient_check
python -m experiments.sigmoid_mse_baseline
python -m experiments.ablation
python -m experiments.architecture_comparison
python -m experiments.final_evaluation
python -m experiments.feature_analysis
# Optional framework baseline (needs scikit-learn):
# python -m experiments.sklearn_baseline
