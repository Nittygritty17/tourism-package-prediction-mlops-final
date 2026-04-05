import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier  # Your best model
from sklearn.metrics import roc_auc_score

# Config - MATCHES your notebook
HFTOKEN = os.getenv("HFTOKEN")
HFDATASETID = "nittygritty2106/tourismdataset"  # FIXED
HFMODELREPO = "nittygritty2106/travelpredictionmlops"
TARGETCOL = "ProdTaken"
MLFLOW_EXPERIMENT = "tourism-package-prediction-experiments"

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Load data
traindf = load_dataset(HFDATASETID, split="train", token=HFTOKEN).to_pandas()
testdf = load_dataset(HFDATASETID, split="test", token=HFTOKEN).to_pandas()

# Clean (from your notebook)
for dftemp in [traindf, testdf]:
    if "Gender" in dftemp.columns:
        dftemp["Gender"] = dftemp["Gender"].replace("Fe Male", "Female")
    cols_to_drop = ["Unnamed: 0", "CustomerID", "__index_level_0__"]
    existing_cols = [col for col in cols_to_drop if col in dftemp.columns]
    if existing_cols:
        dftemp.drop(columns=existing_cols, inplace=True)

Xtrain = traindf.drop(columns=[TARGETCOL])
ytrain = traindf[TARGETCOL].astype(int)
Xtest = testdf.drop(columns=[TARGETCOL])
ytest = testdf[TARGETCOL].astype(int)

# Features
numcols = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
catcols = Xtrain.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Preprocessor (from your notebook)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numcols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), catcols)
    ]
)

# Best model from your experiments
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.5, random_state=42
    ))
])

with mlflow.start_run(run_name="production-gradient-boosting"):
    model.fit(Xtrain, ytrain)
    
    # Evaluate
    y_prob = model.predict_proba(Xtest)[:, 1]
    roc_auc = roc_auc_score(ytest, y_prob)
    mlflow.log_metric("test_roc_auc", roc_auc)
    
    # Save & upload
    joblib.dump(model, "model.joblib")
    mlflow.sklearn.log_model(model, "model")
    
    from huggingface_hub import HfApi
    api = HfApi(token=HFTOKEN)
    api.create_repo(repo_id=HFMODELREPO, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj="model.joblib",
        path_in_repo="model.joblib",
        repo_id=HFMODELREPO,
        repo_type="model",
        token=HFTOKEN
    )
    
    print(f"✅ Model uploaded. Test ROC-AUC: {roc_auc:.4f}")
