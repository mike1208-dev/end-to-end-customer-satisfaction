# End-to-End Customer Satisfaction MLOps Project

Predicts the review score (1–5) a customer will give an order, and wraps the model in a small MLOps
workflow: **ZenML** pipelines for training, a quality gate, the **MLflow Model Registry** for
deployment, batch inference, and a **Streamlit** app that serves the current production model.

## Table of Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Installation and setup](#installation-and-setup)
- [Usage](#usage)
- [Tests](#tests)
- [Results](#results)
- [Notes and known issues](#notes-and-known-issues)
- [Contributing](#contributing)

## How it works

The data is the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
merged into one CSV (`data/olist_customers_dataset.csv`, 115,609 rows). The model uses 12 numeric order
features (payment, price, freight, product size and description) to predict `review_score`.

```
train_pipeline                    (python run_pipeline.py)
  ingest_data → clean_data → train_model → evaluation          metrics logged to MLflow

continuous_deployment_pipeline    (python run_deployment.py --config deploy)
  ingest_data → clean_data → train_model → evaluation
      → deployment_trigger   quality gate: test RMSE < --max-rmse (default 1.35)
      → register_model       if passed: register in the MLflow Model Registry and
                             point the alias "production" at the new version

inference_pipeline                (python run_deployment.py --config predict)
  dynamic_importer (100 sample rows) → predictor (loads models:/customer_satisfaction_model@production)

streamlit_app.py                  (streamlit run streamlit_app.py)
  12 inputs → loads the "production" model → predicted score
```

The default gate of 1.35 means "deploy only if the model beats always predicting the average
review score", which has an RMSE of about 1.38 on this data.

**Tech stack:** Python 3.11 · ZenML 0.98 · MLflow 3.18 · LightGBM / XGBoost / scikit-learn ·
Optuna (optional hyperparameter tuning) · pandas 3 · Streamlit · pytest

## Project structure

| Path | Purpose |
|---|---|
| `model/` | Plain-Python ML logic: data cleaning, model classes (LightGBM, XGBoost, RandomForest, Linear Regression), Optuna tuning, metrics (MSE / RMSE / R²) |
| `steps/` | ZenML steps wrapping that logic: `ingest_data`, `clean_data`, `train_model`, `evaluation` |
| `pipelines/training_pipeline.py` | Training pipeline |
| `pipelines/deployment_pipeline.py` | Deployment pipeline (quality gate + registration) and inference pipeline |
| `pipelines/model_registry.py` | Model name, `production` alias, and helpers to look up / load the production model |
| `run_pipeline.py` | Runs the training pipeline |
| `run_deployment.py` | Runs deployment and/or batch inference (`--config`, `--max-rmse`) |
| `streamlit_app.py` | Web app for single predictions |
| `tests/data_test.py` | pytest suite |
| `requirements.txt` | Direct dependencies, pinned |
| `requirements-lock.txt` | Every installed package with exact versions (`pip freeze`), for an identical environment |

## Installation and setup

Requires **Python 3.11** and Git. Commands are for **PowerShell** on Windows; on macOS/Linux, activate
with `source .venv/bin/activate` instead.

1. Clone the repository and enter it:
   ```powershell
   git clone https://github.com/mike1208-dev/end-to-end-customer-satisfaction.git
   cd end-to-end-customer-satisfaction
   ```

2. Create and activate a virtual environment, then install the dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   For an exact copy of the tested environment, use `pip install -r requirements-lock.txt` instead.

   > If PowerShell says *"running scripts is disabled on this system"*, run
   > `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once, then activate again.

3. Set up ZenML and an MLflow stack (once per machine):
   ```powershell
   zenml init
   zenml integration install mlflow -y
   zenml experiment-tracker register mlflow_tracker --flavor=mlflow
   zenml model-deployer register mlflow --flavor=mlflow
   zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
   zenml stack describe
   ```
   `zenml stack describe` should show `mlflow_stack` as active, with `mlflow_tracker` and `mlflow` in it.

   > `zenml[local]` (in `requirements.txt`) is required: plain `zenml` is only a client for a remote
   > ZenML server and fails locally with `No module named 'sqlalchemy_utils'`.

## Usage

**Train** and log metrics to MLflow:
```powershell
python run_pipeline.py
```

**Deploy:** train, apply the quality gate, and register the model as `production` if it passes:
```powershell
python run_deployment.py --config deploy
python run_deployment.py --config deploy --max-rmse 1.0   # stricter gate: this model is rejected
```

**Batch inference** with the current production model:
```powershell
python run_deployment.py --config predict
```
`--config deploy_and_predict` (the default) runs both.

**Web app** at http://localhost:8501:
```powershell
streamlit run streamlit_app.py
```
Run `--config deploy` first; the app needs a registered `production` model.

**MLflow UI** at http://127.0.0.1:5000. The scripts above print the exact command, which looks like:
```powershell
mlflow ui --backend-store-uri 'sqlite:///C:\Users\<you>\AppData\Roaming\zenml\local_stores\<id>\mlflow.db'
```
Switch the UI to **Model training** mode, then open **Training runs** (metrics, parameters, artifacts)
or **Models** (`customer_satisfaction_model` and its `@production` alias).

**Choosing a model:** `train_model` takes `model_name` (`lightgbm` (default), `xgboost`,
`randomforest`, `linear_regression`) and `fine_tuning` (Optuna search, 100 trials). To change them,
edit the `train_model(...)` call in the pipeline, e.g. `train_model(x_train, x_test, y_train, y_test, model_name="xgboost")`.

## Tests

```powershell
python -m pytest tests -v
```
11 tests (about 5 seconds) covering data cleaning, the train/test split, the feature columns the
Streamlit app sends, the metric classes, and whether a freshly trained model passes the deployment
gate. Expected result: `10 passed, 1 xfailed` (see the order leakage note below).

## Results

LightGBM with default parameters, on a random 80/20 split:

| Metric | Value |
|---|---|
| MSE | 1.739 |
| RMSE | 1.319 |
| R² | 0.083 |

R² is low: order and product features alone explain little of why a customer is (un)happy. Delivery
delays, which are probably the strongest signal, are dropped in cleaning. Treat this project as an
MLOps workflow example, not as an accurate predictor.

## Notes and known issues

- **Deployment uses the MLflow Model Registry, not a prediction server.** ZenML's MLflow model
  deployer runs the server as a background daemon, which is not supported on Windows. The deployment
  pipeline therefore registers the model with a `production` alias, and the app and the inference
  pipeline load it from the registry. Every ZenML command prints
  `Daemon functionality is currently not supported on Windows.`; this is harmless.
- **Order-level leakage.** An order can have several rows (one per payment or item). The random row
  split puts about 21% of test orders in the training set as well, so test scores are somewhat
  optimistic. The fix is to split by `order_id` (e.g. `GroupShuffleSplit`). `tests/data_test.py`
  tracks this as an expected failure.
- **Hyperparameter tuning scores on the test set.** `HyperparameterTuner` picks parameters by test
  score, which biases the reported metrics when `fine_tuning=True`. A separate validation split
  would avoid this.
- **pandas 3 (Copy-on-Write)** silently ignores `df["col"].fillna(..., inplace=True)`. The cleaning
  code assigns the result instead: `df[col] = df[col].fillna(...)`.
- **Model saving:** MLflow saves models with skops, which only loads types it is told to trust. The
  `register_model` step passes the exact types of the trained model (`skops_trusted_types`).

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.
