# Final Experiment Results

Use `v2_results_second_run/v2/tables/` as the complete final experiment run.
This directory contains the full v2 protocol outputs:

- tabular HPO for Logistic Regression, Random Forest, and XGBoost
- deep-learning HPO and multi-seed refit
- leakage probe
- sensor-source ablation
- final held-out test evaluation
- bootstrap confidence intervals and summary analysis

The later local rerun was used only to check local execution. It was stopped
because full local sensor ablation was taking too long on CPU. The final report
and slides should use the complete saved second-run outputs instead.

## Primary Selection Rule

The project selects candidates using validation PR-AUC. Accuracy is not used for
model selection because asthma exacerbation events are rare.

## Final Held-Out Test Summary

From `v2_results_second_run/v2/tables/final_test_v2_summary.csv`:

| Family | Model | T | History | Washout | Test PR-AUC | Test ROC-AUC | F1 | Tuned F1 | Test N |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tabular | Random Forest | 4 | 14 days | 14 days | 0.555 | 0.934 | 0.000 | 0.667 | 325 |
| Tabular | XGBoost | 3 | 14 days | 14 days | 0.526 | 0.875 | 0.308 | 0.067 | 324 |
| Tabular | Logistic Regression | 4 | 14 days | 0 days | 0.022 | 0.615 | 0.041 | 0.017 | 350 |
| Deep learning | RNN | 3 | 14 days | 14 days | 0.261 +/- 0.148 | 0.902 +/- 0.057 | 0.000 | n/a | 324 per seed |

Important caution: the held-out test split has only four positive events for
the best RF/XGB/RNN configurations. The ranking signal is promising, but the
confidence intervals are very wide, so the model is not clinically deployable.

## Confidence Intervals

The best Random Forest PR-AUC has a 95% bootstrap interval of approximately
0.033 to 1.000. This is wide because the positive test count is very small.

Interpretation:

- RF and XGB found stronger rank signal than LR.
- The apparent RF advantage is not statistically stable enough for deployment.
- High ROC-AUC should not be over-interpreted without PR-AUC, recall, threshold
  behavior, and calibration.

## Leakage Probe

The leakage probe summary contains 81 configurations. The overall shuffled-label
validation ROC-AUC mean is approximately 0.505, which is close to random.

Interpretation:

- There is no strong evidence of systematic leakage across the overall protocol.
- Some individual configurations vary strongly under label shuffling, so unstable
  configurations should be interpreted cautiously.

## Sensor Ablation

The complete sensor ablation file contains 729 rows:

- 243 Logistic Regression rows
- 243 Random Forest rows
- 243 XGBoost rows

Interpretation:

- Multimodal/all-sensor tabular models produced the strongest headline result.
- Ablation patterns were not monotonic, so exact modality importance is not
  cleanly separable from this small dataset.
- Questionnaire-derived predictors may be close to symptom-based labels and
  should be discussed as a possible proxy-label risk.

## Final Scientific Conclusion

The project demonstrates a full patient-safe early-warning pipeline for asthma
exacerbation prediction using multimodal AAMOS-00 signals. The best validation-
selected held-out model was Random Forest with 14 days of history and 14 days of
post-event washout. Results suggest a useful signal, but the small held-out
positive count and wide confidence intervals mean the system is a research
prototype, not a clinically deployable alert model.
