# Early-Warning Prediction of Asthma Exacerbations

This source-code package contains the reproducible project code for predicting
upcoming asthma exacerbation risk from AAMOS-00 wearable and digital-health
signals.

## What Is Included

- `src/`: reusable data cleaning, labeling, feature engineering, modeling, and
  evaluation modules.
- `scripts/`: command-line experiment runners.
- `scripts/v2/`: event-episode v2 protocol, including tabular HPO, deep-learning
  HPO, leakage probes, sensor ablation, final test evaluation, and analysis.
- `notebooks/`: Colab/local notebooks used to run the v2 protocol.
- `requirements.txt`: local development dependencies.
- `requirements-colab.txt`: pinned Colab/GPU dependencies.
- `V2_METRIC_PROTOCOL_RUNBOOK.md`: run instructions for the final v2 protocol.

Raw data and generated outputs are intentionally not included in this source
bundle.

## Reproducible Run Order

From the project root:

```powershell
python scripts\v2\run_metric_protocol.py check
python scripts\v2\run_metric_protocol.py local-tabular --reuse-features
python scripts\v2\run_metric_protocol.py local-gates
python scripts\v2\run_metric_protocol.py finalize
```

Use Colab/GPU for deep-learning sweeps:

```python
!python scripts/v2/run_metric_protocol.py colab-dl
```

For a complete Colab run:

```python
!python scripts/v2/run_metric_protocol.py full-v2
```

## Selection Discipline

Model and representation selection uses validation PR-AUC as the primary metric.
The test set is used only after validation selection is complete. Accuracy is
reported only as secondary context because asthma exacerbation windows are rare.

## Important Note

The `--reuse-features` flag is designed to reuse existing v2 sample and feature
parquets without rebuilding intermediate Phase-3-style artifacts. If required
intermediate parquets are missing, the runner fails loudly instead of silently
overwriting them.
