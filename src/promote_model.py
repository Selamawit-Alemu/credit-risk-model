import mlflow

client = mlflow.tracking.MlflowClient()

model_name = "CreditRiskRandomForestModel"
version ="8"

client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production",
    archive_existing_versions=True  # Optional: archives any existing prod versions
)

print(f"Model '{model_name}' version {version} has been promoted to Production stage.")
