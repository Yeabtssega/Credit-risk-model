import mlflow

def promote_model(model_name="best_model", version=1):
    client = mlflow.MlflowClient()
    # Transition model version stage to "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model {model_name} version {version} promoted to Production.")

if __name__ == "__main__":
    promote_model()
