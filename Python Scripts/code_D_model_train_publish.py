# code_D_model_train_publish.py
# Purpose: Load TRAIN from CAS, train sklearn models,
# evaluate, score, generate reports, export to SAS Model Manager.
# ======================================================

import os
import joblib
import requests
import swat
# import boto3
import json
import getpass
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sasctl import Session, current_session
from sasctl.services import model_repository as mr
from sasctl.services import projects as projects_svc
import sasctl.pzmm as pzmm

# ------------------------------------------------------
# 1. Load data from S3 using boto3
# ------------------------------------------------------

# # Replace with your S3 bucket name and file key
# bucket_name = "indiadatalake"
# file_key = "HDFC_POC_TRAIN.csv" # or .json, .parquet, etc.

# # Create an S3 client
# s3 = boto3.client('s3')

# # Get the object from S3
# obj = s3.get_object(Bucket=bucket_name, Key=file_key)

# df = pd.read_csv(io.BytesIO(obj['Body'].read()))
conn = swat.CAS("sas-cas-server-default-client", 5570, "Demo1", "Password1")

train_tbl = conn.CASTable("HDFC_POC_TRAIN", caslib="PYS3")

df = train_tbl.to_frame()

print("Training data loaded from CAS:", df.shape)

# ------------------------------------------------------
# 2. Prepare X, y
# ------------------------------------------------------

target = ['Response', 'salary_band_flag', 'vintage_bucket']
X = df.drop(columns=target, errors='ignore')
y = df['Response']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------
# 3. Define models
# ------------------------------------------------------

models = {
    'Logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(eval_metric='auc', random_state=42, use_label_encoder=False)
}

# ------------------------------------------------------
# 4. Preprocessing
# ------------------------------------------------------

num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

logistic_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', 'passthrough', cat_cols)
])

# ------------------------------------------------------
# 5. Evaluation
# ------------------------------------------------------

def evaluate_model(dataset_name, model_name, y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)

    df_eval = pd.DataFrame({'y': y_true, 'score': y_pred_proba}).sort_values('score', ascending=False)
    df_eval['cum_event'] = np.cumsum(df_eval['y']) / df_eval['y'].sum()
    df_eval['cum_non_event'] = np.cumsum(1 - df_eval['y']) / (len(df_eval) - df_eval['y'].sum())
    ks = max(abs(df_eval['cum_event'] - df_eval['cum_non_event']))
    lift = df_eval.head(len(df_eval)//10)['y'].mean() / df_eval['y'].mean()

    return {
        'Dataset': dataset_name,
        'Model': model_name,
        'AUC': round(auc, 4),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'KS': round(ks, 4),
        'Lift@Top10%': round(lift, 2)
    }, y_pred

# ------------------------------------------------------
# 6. Train + Evaluate
# ------------------------------------------------------

results = []
trained_models = {}

for name, model in models.items():
    preprocessor = logistic_preprocessor if name == 'Logistic' else ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', 'passthrough', cat_cols)
    ])

    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict_proba(X_val)[:, 1]
    trained_models[name] = pipe

    res, _ = evaluate_model("Validation", name, y_val, pred)
    results.append(res)

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\n=== Validation Metrics ===\n")
print(results_df)

# ------------------------------------------------------
# 7. Save Champion Model
# ------------------------------------------------------

best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

os.makedirs("../Models", exist_ok=True)
joblib.dump(best_model, "../Models/champion_model.pkl")
print("\nChampion model saved:", best_model_name)

#=======================
# AUTHENTICATION
#=======================
username = "Demo1"
password = "Password1"
host     = "sasviyaind.sas.com"

sess = Session(host, username=username, password=password, verify_ssl=False)
current_session(sess)
print("\nAuthenticated & session active.")


def safe_proba_output(proba_array):
    # If predict_proba returns only probability of class "1"
    if proba_array.ndim == 1:
        # Reconstruct 2-column proba array: [P(class0), P(class1)]
        proba_array = np.column_stack([1 - proba_array, proba_array])
    return proba_array

# ------------------------------------------------------
# CONFIG (adapted)
# ------------------------------------------------------
REPOSITORY = None  # None -> will use default repository
PROJECT_NAME = "HDFC_POC_Project"
BASE_EXPORT_DIR = Path("../Models/MM_Export_sasctl")
BASE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

predictor_columns = list(X.columns)
target_column = "Response"
score_metrics = ["EM_CLASSIFICATION", "EM_EVENTPROBABILITY"]
target_values = ["0", "1"]
target_index = 1  # Index of positive class (1) in target_values

# Helper: ensure project exists (unchanged)
def ensure_project(name, repository=None, description=None, function="classification"):
    repo = mr.default_repository() if repository is None else mr.get_repository(repository)
    try:
        proj = mr.get_project(name)
        if proj:
            print(f"Using existing project: {name}")
            return proj
    except Exception:
        pass
    project_payload = {
        "name": name,
        "description": description or f"Project created by python script on {datetime.utcnow().isoformat()}",
        "function": function
    }
    created = mr.create_project(project_payload, repo)
    print(f"Created project: {name}")
    return created

# Create/get project
project = ensure_project(PROJECT_NAME, repository=REPOSITORY)

# ------------------------------------------------------
# Adapted pzmm Functions (mirroring working code)
# ------------------------------------------------------
def write_json_files(data, predict, target, path, prefix, description, algorithm, modeler):
    # Write input variable mapping to a json file
    pzmm.JSONFiles.write_var_json(input_data=data[predict], is_input=True, json_path=path)
  
    # Set output variables and assign an event threshold, then write output variable mapping
    output_var = pd.DataFrame(columns=score_metrics, data=[["1", 0.5]])  # Example: positive class (nominal), probability (interval)
    pzmm.JSONFiles.write_var_json(output_var, is_input=False, json_path=path)
  
    # Write model properties to a json file
    pzmm.JSONFiles.write_model_properties_json(
        model_name=prefix,
        target_variable=target,  # Target variable to make predictions about (Response in this case)
        target_values=target_values,  # Possible values for the target variable (0 or 1 for binary classification)
        json_path=path,
        model_desc=description,
        model_algorithm=algorithm,
        modeler=modeler,
    )
  
    # Write model metadata to a json file so that SAS Model Manager can properly identify all model files
    pzmm.JSONFiles.write_file_metadata_json(model_prefix=prefix, json_path=path)

def write_model_stats(x_train, y_train, test_predict, test_proba, y_test, model, path, prefix):
    # Calculate train predictions
    train_predict = model.predict(x_train)    
    train_proba = safe_proba_output(model.predict_proba(x_train))[:, 1]

    # Assign data to lists of actual and predicted values
    train_data = pd.concat([
        y_train.reset_index(drop=True),
        pd.Series(train_predict),
        pd.Series(train_proba)
    ], axis=1)

    test_data = pd.concat([
        y_test.reset_index(drop=True),
        pd.Series(test_predict),
        pd.Series(test_proba)
    ], axis=1)

    # Calculate the model statistics, ROC chart, and Lift chart; then write to json files
    pzmm.JSONFiles.calculate_model_statistics(
        target_value=1, 
        train_data=train_data, 
        test_data=test_data, 
        json_path=path
    )

    full_training_data = pd.concat([y_train.reset_index(drop=True), x_train.reset_index(drop=True)], axis=1)

    pzmm.JSONFiles.generate_model_card(
        model_prefix=prefix,
        model_files = path,
        algorithm = str(type(model).__name__),
        train_data = full_training_data,
        train_predictions=train_predict,
        target_type='classification',
        target_value=1,
        interval_vars=predictor_columns,
        selection_statistic='_RASE_',
    )

# ------------------------------------------------------
# 9. SAS MODEL MANAGER EXPORT USING PZMM
# ------------------------------------------------------
published_models = {}
for model_name, model_pipe in trained_models.items():
    safe_name = model_name.replace(" ", "_")
    export_dir = BASE_EXPORT_DIR / safe_name
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Serialize the model to pickle format using pzmm
    pzmm.PickleModel.pickle_trained_model(
        model_prefix=safe_name,
        trained_model=model_pipe,
        pickle_path=export_dir
    )
    print(f"Pickled model: {export_dir / f'{safe_name}.pickle'}")
    
    # 2) Determine metadata
    algorithm = type(model_pipe.named_steps['clf']).__name__ if 'clf' in model_pipe.named_steps else str(type(model_pipe))
    description = f"{model_name} trained {datetime.utcnow().isoformat()}"
    modeler = 'Demo1'  # Customize as needed
    
    # 3) Write JSON files using pzmm (variables, properties, metadata)
    write_json_files(X, predictor_columns, target_column, export_dir, safe_name, description, algorithm, modeler)
    print(f"Wrote JSON metadata files: {export_dir}")
    
    # 4) Calculate model statistics using pzmm
    train_predict = model_pipe.predict(X_train)
    train_proba = model_pipe.predict_proba(X_train)[:, 1]
    test_predict = model_pipe.predict(X_val)
    test_proba = model_pipe.predict_proba(X_val)[:, 1]
    write_model_stats(X_train, y_train, test_predict, test_proba, y_val, model_pipe, export_dir, safe_name)
    print(f"Wrote model statistics: {export_dir}")
    
    # 5) Import model using pzmm (generates score code automatically)
    pzmm.ScoreCode.score_code = ""  # Reinitialize for each model
    pzmm.ImportModel.import_model(
        model_files=export_dir,  # Where are the model files?
        model_prefix=safe_name,  # What is the model name?
        project=PROJECT_NAME,  # What is the project name?
        input_data=X,  # What does example input data look like?
        predict_method=[model_pipe.predict_proba, [int, float]],  # Predict method and output dtypes (class: int, prob: float)
        score_metrics=score_metrics,  # What are the output variables?
        overwrite_model=True,  # Overwrite the model if it already exists?
        target_values=target_values,  # What are the expected values of the target variable?
        target_index=target_index,  # What is the index of the target value in target_values?
        model_file_name=f"{safe_name}.pickle",  # How was the model file serialized?
        missing_values=True  # Does the data include missing values?
    )
    print(f"Imported model into SAS Model Manager: {safe_name}")
    
    # Track published model
    published_models[model_name] = {
        "export_dir": str(export_dir),
        "safe_name": safe_name
    }

# Summary
print("\nPublishing summary:")
for k, v in published_models.items():
    print(f" - {k}: dir={v['export_dir']}")
print("\nDone. Models published. Champion set to:", best_model_name)
