"""Where the production model lives: the MLflow Model Registry used by ZenML's MLflow tracker."""
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

MODEL_NAME = "customer_satisfaction_model"
MODEL_ALIAS = "production"


def get_production_version():
    """Returns the registry version currently marked 'production', or None."""
    try:
        return MlflowClient(get_tracking_uri()).get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    except MlflowException:
        return None


def load_production_model():
    """Loads the 'production' model from the registry as an MLflow pyfunc model."""
    mlflow.set_tracking_uri(get_tracking_uri())
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
