import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from pipelines.model_registry import MODEL_ALIAS, MODEL_NAME, get_production_version

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Run only the deployment pipeline (`deploy`), only batch prediction "
    "against the production model (`predict`), or both (`deploy_and_predict`).",
)
@click.option(
    "--max-rmse",
    default=1.35,
    help="Maximum test RMSE a model may have to be promoted to production.",
)
def main(config: str, max_rmse: float):
    """Train, gate and register a model, and/or run batch predictions with it."""
    if config in (DEPLOY, DEPLOY_AND_PREDICT):
        continuous_deployment_pipeline(max_rmse=max_rmse)

    version = get_production_version()
    if config in (PREDICT, DEPLOY_AND_PREDICT):
        if version is None:
            print("[red]No production model registered yet; skipping prediction. "
                  "Run with `--config deploy` first.[/red]")
        else:
            inference_pipeline()

    print(
        "You can run:\n"
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n"
        "...to inspect runs and the registered model in the MLflow UI."
    )
    if version is not None:
        print(f"Production model: [bold]{MODEL_NAME}[/bold] version {version} (alias '{MODEL_ALIAS}').")


if __name__ == "__main__":
    main()
