import mlflow

client = mlflow.tracking.MlflowClient()

experiment_id = "591355787569453947"  # Credit Risk Model experiment ID
runs = client.search_runs(experiment_ids=[experiment_id])

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    artifacts = client.list_artifacts(run.info.run_id)
    if artifacts:
        for artifact in artifacts:
            print(f"  Artifact path: {artifact.path}")
    else:
        print("  No artifacts found")
