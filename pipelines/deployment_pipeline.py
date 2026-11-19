import io

import mlflow
import numpy as np
import pandas as pd
import skops.io
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import RegressorMixin
from zenml import pipeline, step
from zenml.client import Client

from pipelines.model_registry import MODEL_ALIAS, MODEL_NAME, load_production_model
from pipelines.utils import get_data_for_test
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False)
def dynamic_importer() -> str:
    """Loads a sample of 100 cleaned rows (as JSON) to run batch predictions on."""
    return get_data_for_test()


@step
def deployment_trigger(rmse: float, max_rmse: float) -> bool:
    """Quality gate: only deploy if the model's test RMSE is below max_rmse."""
    return rmse < max_rmse


@step(experiment_tracker=experiment_tracker.name)
def register_model(deploy_decision: bool, model: RegressorMixin, x_test: pd.DataFrame) -> None:
    """Registers the model in the MLflow Model Registry and points the 'production'
    alias at it, but only if it passed the quality gate."""
    if not deploy_decision:
        print("Model did not pass the quality gate; production model unchanged.")
        return
    x_sample = x_test.head(100).astype("float64")
    # MLflow saves sklearn models with skops, which only loads types it is told to trust.
    # Non-sklearn estimators (LightGBM, XGBoost) need their types listed; we trust exactly
    # the types of the model this pipeline just trained.
    trusted_types = skops.io.get_untrusted_types(data=skops.io.dumps(model))
    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(x_sample, model.predict(x_sample)),
        registered_model_name=MODEL_NAME,
        skops_trusted_types=trusted_types,
    )
    version = model_info.registered_model_version
    MlflowClient().set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
    print(f"Registered {MODEL_NAME} version {version} as '{MODEL_ALIAS}'.")


@step(enable_cache=False)
def predictor(data: str) -> np.ndarray:
    """Runs batch predictions with the current production model."""
    df = pd.read_json(io.StringIO(data), orient="split").astype("float64")
    model = load_production_model()
    return np.asarray(model.predict(df))


@pipeline(enable_cache=False)
def continuous_deployment_pipeline(max_rmse: float = 1.35):
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)
    deployment_decision = deployment_trigger(rmse=rmse, max_rmse=max_rmse)
    register_model(deployment_decision, model, x_test)


@pipeline(enable_cache=False)
def inference_pipeline():
    batch_data = dynamic_importer()
    predictor(batch_data)
