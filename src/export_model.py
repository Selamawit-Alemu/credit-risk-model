import joblib
from pathlib import Path
import mlflow.sklearn

# Replace with your actual run ID
run_id = "05ebc4840e8345f0ba83d7fffbf96b20"

# Load model from MLflow run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/random_forest_model")

# Create models directory if it doesn't exist
Path("models/").mkdir(exist_ok=True)

# Export model to models/
joblib.dump(model, "models/credit_model.pkl")

print("✅ Model exported to models/credit_model.pkl")
pipeline = joblib.load("preprocessing_pipeline.pkl")
joblib.dump(pipeline, "models/preprocessing_pipeline.pkl")