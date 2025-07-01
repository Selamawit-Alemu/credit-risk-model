import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

# Correct imports for feature engineering and preprocessing pipeline
from feature_engineering import get_feature_engineering_pipeline

# Add missing imports for proxy target variable creation (RFM)
from proxy_target import calculate_rfm, scale_rfm, perform_kmeans_clustering

# ---------------- Load and Preprocess Data ----------------
df = pd.read_csv("data/raw/data.csv")
df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

# --- Generate RFM Target ---
snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
rfm = calculate_rfm(df, snapshot_date)
rfm_scaled, _ = scale_rfm(rfm)
rfm["cluster"], _ = perform_kmeans_clustering(rfm_scaled)
high_risk_cluster = rfm.groupby("cluster")[["frequency", "monetary"]].mean().sort_values(["frequency", "monetary"]).index[0]
rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
df = df.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")
df["is_high_risk"] = df["is_high_risk"].fillna(0).astype(int)

# ---------------- Prepare Features and Target ----------------
X = df.drop(columns=["is_high_risk"])
y = df["is_high_risk"]

# ---------------- Full Feature Engineering and Preprocessing Pipeline ----------------
full_pipeline = get_feature_engineering_pipeline()
X_processed = full_pipeline.fit_transform(X)

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# ---------------- MLflow Tracking ----------------
mlflow.set_experiment("credit-risk-model")

with mlflow.start_run():
    # --- Model Training with GridSearch ---
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    roc_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

    # --- Log Parameters and Metrics ---
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(best_rf, "random_forest_model")

    # --- Print Evaluation ---
    print("\n🌲 Random Forest - Best Estimator")
    print(classification_report(y_test, y_pred_rf))
    print("ROC AUC:", roc_auc)

    # --- Optional Logistic Regression ---
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    roc_auc_lr = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

    mlflow.log_metric("roc_auc_logreg", roc_auc_lr)
    mlflow.sklearn.log_model(logreg, "logistic_regression_model")

    print("\n📊 Logistic Regression")
    print(classification_report(y_test, y_pred_lr))
    print("ROC AUC:", roc_auc_lr)
    
with mlflow.start_run():
    # ... after training and logging model ...
    
    mlflow.sklearn.log_model(best_rf, "random_forest_model")
    
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"
    model_registration_name = "CreditRiskRandomForestModel"
    
    mlflow.register_model(model_uri, model_registration_name)
