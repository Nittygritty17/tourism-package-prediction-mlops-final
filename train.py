import os
import joblib
from datasets import load_dataset
from huggingface_hub import HfApi, upload_file
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

HFTOKEN = os.getenv("HFTOKEN")
HFDATASETID = "nittygritty2106/travel-prediction-dataset"
HFMODELREPO = "nittygritty2106/travelpredictionmlops"
TARGETCOL = "ProdTaken"

traindf = load_dataset(HFDATASETID, split="train", token=HFTOKEN).to_pandas()
testdf = load_dataset(HFDATASETID, split="test", token=HFTOKEN).to_pandas()

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

numcols = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
catcols = Xtrain.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(Xtrain, ytrain)
joblib.dump(model, "model.joblib")

api = HfApi(token=HFTOKEN)
api.create_repo(repo_id=HFMODELREPO, repo_type="model", exist_ok=True)

upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=HFMODELREPO,
    repo_type="model",
    token=HFTOKEN
)

print("Model uploaded successfully.")
