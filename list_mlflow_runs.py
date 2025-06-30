import mlflow

client = mlflow.tracking.MlflowClient()

experiments = client.search_experiments()  # Use search_experiments() instead of list_experiments()
print("Experiments found:")
for exp in experiments:
    print(f"ID: {exp.experiment_id}, Name: {exp.name}")

if experiments:
    runs = client.search_runs(experiment_ids=[experiments[0].experiment_id])
    print(f"Runs in experiment '{experiments[0].name}':")
    for run in runs:
        print(f"Run ID: {run.info.run_id}, Status: {run.info.status}")
else:
    print("No experiments found.")
