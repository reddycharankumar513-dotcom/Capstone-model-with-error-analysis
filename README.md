# Tuned Churn Model and Error Analysis

This repository adds hyperparameter tuning, held-out evaluation, grouped error analysis, and a deployment verdict to the leak-free pipeline.

## Files
- `tuned_model_error_analysis.ipynb`: reproducible experiment.
- `customer_churn.csv`: synthetic binary classification dataset.
- `tuned_churn_model.joblib`: fitted scikit-learn pipeline.
- `test_predictions_and_errors.csv`: held-out predictions and FP/FN labels.
- `grouped_error_analysis.csv`: error counts/rates by contract, plan, gender, and support-call count.
- `experiment_results.json`: metric summary.
- `requirements.txt`: dependencies.

## Results from this fixed split

| Metric | Baseline | Tuned |
|---|---:|---:|
| Mean CV ROC-AUC (3-fold) | 0.6487 | 0.6501 |
| Test accuracy | 0.6333 | 0.6417 |
| Test ROC-AUC | 0.6840 | 0.6843 |
| Test F1 | 0.7179 | 0.6950 |

Tuning slightly increases test accuracy and ROC-AUC on this split, while F1 falls. This is a trade-off, not an unqualified win. The selected model is chosen using training-only CV ROC-AUC.

## Error analysis

The notebook groups false positives and false negatives by `contract`, `plan`, `gender`, and `support_calls`. False positives may cause unnecessary retention activity; false negatives miss churn-risk customers. Group error rates on this small sample are descriptive and noisy, not causal.

## Verdict

**Do not deploy to real customers yet.** The dataset is synthetic and the measured improvement is small. Before production, evaluate on representative real data with a temporal holdout, choose the decision threshold using business costs, assess calibration and subgroup errors with adequate sample sizes, and monitor drift in a controlled pilot.

## Run it

Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
jupyter notebook
```

Open `tuned_model_error_analysis.ipynb` and run all cells. The dataset is local, so no download is required.

## Leakage controls

The split happens before tuning. Imputation, scaling, and one-hot encoding are in the pipeline. `GridSearchCV` fits the full pipeline independently on each CV training fold. The test split is not used to select hyperparameters.
