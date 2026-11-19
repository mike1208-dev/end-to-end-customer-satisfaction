from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    train_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "Your runs are tracked in the `train_pipeline` experiment, where you can "
        "compare runs side by side."
    )
