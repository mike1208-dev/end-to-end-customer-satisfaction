from zenml import pipeline

from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model


@pipeline(enable_cache=False)
def train_pipeline():
    """Training pipeline: ingest -> clean -> train -> evaluate.

    Metrics (r2_score, rmse) are logged to MLflow by the evaluation step.
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)
