# V2 Metric Protocol Runbook

This runbook is the launch checklist for the current v2 asthma event-episode
experiments.  The selection rule is validation PR-AUC first, with recall,
train-validation gap, Brier score, and simplicity used as tie-breakers.

The runner is:

```powershell
python scripts\v2\run_metric_protocol.py <profile>
```

Use `--dry-run` any time you want to see exactly what will run without starting
training.

## What To Run Locally

Local is best for code checks, feature-table generation, tabular classical
models, leakage probes, and report generation.

```powershell
python scripts\v2\run_metric_protocol.py check
python scripts\v2\run_metric_protocol.py local-tabular
python scripts\v2\run_metric_protocol.py local-gates
python scripts\v2\run_metric_protocol.py finalize
```

For a tiny end-to-end smoke test before a real run:

```powershell
python scripts\v2\run_metric_protocol.py smoke --skip-leakage-gate --analysis-n-boot 100
```

The smoke command uses only `T=3`, `L=7`, `W=7`, logistic regression, and a
small leakage/bootstrap setting.  It is for checking that the pipeline still
runs; it is not for reporting final results.

## What To Run On Colab

Use Colab for the heavy deep-learning sweep and full protocol runs.  Before
running from Colab, commit and push the local branch so the Colab notebook pulls
the same code.

Push source files only.  Do not commit generated `data/processed/v2` parquets or
local `outputs/v2/tables` results unless you intentionally want to version those
artifacts.

Open:

```text
notebooks/10_event_v2_colab.ipynb
```

Then run the setup cells, connect to a GPU runtime, and run either:

```python
!python scripts/v2/run_metric_protocol.py colab-dl
```

or, for the complete protocol on Colab:

```python
!python scripts/v2/run_metric_protocol.py full-v2
```

Use this first if you want to inspect the commands:

```python
!python scripts/v2/run_metric_protocol.py full-v2 --dry-run
```

### Exact Colab Checklist

1. Confirm the branch in the first notebook cell is:

   ```python
   BRANCH = "codex/tarek-event-episode-tuning"
   ```

2. In Colab, choose `Runtime > Change runtime type > GPU`.

3. Run the clone/pull cell.  It should check out
   `codex/tarek-event-episode-tuning`.

4. Run the dependency install cell:

   ```python
   !pip install -q -r requirements-colab.txt
   ```

5. Run the GPU check cell.  It should print `CUDA: True`.

6. Mount Drive and confirm that the dataset folder exists at:

   ```text
   /content/drive/MyDrive/AAMOS/dataset
   ```

   If your dataset folder is elsewhere, edit `data_dir` in the notebook before
   continuing.

7. Run this dry run first:

   ```python
   !python scripts/v2/run_metric_protocol.py full-v2 --dry-run
   ```

8. Start the full heavy protocol:

   ```python
   !python scripts/v2/run_metric_protocol.py full-v2
   ```

9. When the run finishes, run the copy-to-Drive cell.  It copies Colab results
   to:

   ```text
   /content/drive/MyDrive/AAMOS/outputs/v2
   ```

10. Bring the result CSVs back into the local repo before final interpretation,
    especially:

    - `tune_tabular_v2_trials.csv`
    - `tune_tabular_v2_best.csv`
    - `tune_dl_v2_trials.csv`
    - `tune_dl_v2_best.csv`
    - `tune_dl_v2_multiseed.csv`
    - `tune_dl_v2_multiseed_summary.csv`
    - `leakage_probe_v2_summary.csv`
    - `sensor_ablation_v2.csv`
    - `final_test_v2_summary.csv`
    - `final_test_v2_predictions_tabular.csv`
    - `final_test_v2_predictions_dl.csv`, if produced

## Profiles

| Profile | Intended machine | What it does |
| --- | --- | --- |
| `check` | Local | Compiles `src/event_v2` and `scripts/v2`. |
| `local-tabular` | Local | Runs LR, Random Forest, and XGBoost tabular HPO. |
| `local-gates` | Local | Runs leakage probe and sensor ablation after tabular winners exist. |
| `colab-dl` | Colab/GPU | Runs GRU, LSTM, RNN, and CNN HPO plus multi-seed winner refits. |
| `finalize` | Local or Colab | Runs final validation-selected test evaluation and offline analysis. |
| `full-v2` | Colab/GPU | Runs tabular, DL, leakage, ablation, final test, and analysis. |
| `smoke` | Local | Quick small-grid pipeline check. |

## Default Experiment Grid

Unless overridden, the real protocol uses:

- Event thresholds: `2,3,4`
- History lengths: `3,7,14` days
- Washouts: `0,7,14` days
- Tabular models: `lr,rf,xgb`
- DL models: `gru,lstm,rnn,cnn`
- DL epochs: `30`
- DL batch size: `32`
- DL early-stopping patience: `5`
- DL multi-seeds: `42,43,44,45,46`
- Leakage shuffles: `5`

Useful overrides:

```powershell
python scripts\v2\run_metric_protocol.py local-tabular --thresholds 3 --lengths 7 --washouts 7 --algos lr,rf
python scripts\v2\run_metric_protocol.py colab-dl --archs gru,lstm --epochs 20
python scripts\v2\run_metric_protocol.py full-v2 --quick
```

## Expected Outputs

The main outputs are written under:

```text
outputs/v2/tables/
```

Important files:

- `tune_tabular_v2_trials.csv`
- `tune_tabular_v2_best.csv`
- `tune_dl_v2_trials.csv`
- `tune_dl_v2_best.csv`
- `tune_dl_v2_multiseed.csv`
- `tune_dl_v2_multiseed_summary.csv`
- `leakage_probe_v2_summary.csv`
- `sensor_ablation_v2.csv`
- `final_test_v2_results.csv`
- `final_test_v2_predictions_tabular.csv`
- `final_test_v2_predictions_dl.csv`
- `analysis_v2_tabular_ci.csv`
- `analysis_v2_dl_ci.csv`

## Selection Discipline

Pick the winner from validation metrics only.  The test set is touched only by
`finalize` or `full-v2` after the validation winner has already been selected.

Accuracy is reported only as secondary context.  It must not choose the model,
because the positive asthma-event class is rare.

If a raw-minute or high-detail representation is added later, it should be
plugged into this same protocol and judged by the same validation PR-AUC,
recall, calibration, train-validation gap, and leakage-probe rules.
