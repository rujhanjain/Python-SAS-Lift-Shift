# code_B_model_score.py
# Purpose: Load TRAIN from CAS, train sklearn models,
# evaluate, score, generate reports, export to SAS Model Manager.
# ======================================================

import os
print(f'Current Working Directory: {os.getcwd()}')
import joblib
import requests
import swat
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

import sasctl.pzmm as pzmm
from sasctl import Session

# ------------------------------------------------------
# 1. Load data from CAS using SWAT
# ------------------------------------------------------

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

# ------------------------------------------------------
# 8. Connect to sasctl for session
# ------------------------------------------------------

hostname = "sasviyaind.sas.com"

url = f"https://{hostname}/SASLogon/oauth/token"
authBody = 'grant_type=authorization_code&code=%s' %('1gyZg0RDEbTNuEMW7mJIPq2HpK67YUvV')
headersAuth={'Accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded'}
r =  requests.request('POST', url, data= authBody, headers=headersAuth, auth=('sas.cli', ''),verify=False)
print(r)
token = r.json()['access_token']
print(token)

mm_session = Session(
    hostname, token
)
print("SASCTL session created for Model Manager export.")

# ------------------------------------------------------
# 9. SAS MODEL MANAGER EXPORT
# ------------------------------------------------------

model_prefix = best_model_name.replace(" ", "_")
base_folder = Path("../Models/MM_Export")
export_path = base_folder / model_prefix
export_path.mkdir(parents=True, exist_ok=True)

# 9a. Pickle in pzmm format
pzmm.PickleModel.pickle_trained_model(
    model_prefix=model_prefix,
    trained_model=best_model,
    pickle_path=export_path
)

# 9b. Write variable JSON
pzmm.JSONFiles.write_var_json(
    input_data=X,
    is_input=True,
    json_path=export_path
)

output_var = pd.DataFrame(columns=["EM_CLASSIFICATION", "EM_EVENTPROBABILITY"], data=[["A", 0.5]])
pzmm.JSONFiles.write_var_json(
    output_var,
    is_input=False,
    json_path=export_path
)

# 9c. Write model properties
pzmm.JSONFiles.write_model_properties_json(
    model_name=model_prefix,
    target_variable="Response",
    target_values=["0", "1"],
    json_path=export_path,
    model_desc=f"Champion model {model_prefix}",
    model_algorithm=best_model.__class__.__name__,
    modeler="python_user"
)

# 9d. File metadata JSON
pzmm.JSONFiles.write_file_metadata_json(
    model_prefix=model_prefix,
    json_path=export_path
)

# 9e. Model Statistics + Model Card
# Need train-prediction and test-prediction
train_pred = best_model.predict(X_train)
train_proba = best_model.predict_proba(X_train)[:, 1]
test_pred = best_model.predict(X_val)
test_proba = best_model.predict_proba(X_val)[:, 1]

train_data = pd.DataFrame({"actual": y_train, "pred": train_pred, "proba": train_proba})
test_data = pd.DataFrame({"actual": y_val, "pred": test_pred, "proba": test_proba})

pzmm.JSONFiles.calculate_model_statistics(
    target_value=1,
    train_data=train_data,
    test_data=test_data,
    json_path=export_path
)

full_training_data = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)

pzmm.JSONFiles.generate_model_card(
    model_prefix=model_prefix,
    model_files=export_path,
    algorithm=str(type(best_model.named_steps["clf"]).__name__),
    train_data=full_training_data,
    train_predictions=train_pred,
    target_type='classification',
    target_value=1,
    interval_vars=num_cols,
    selection_statistic='_RASE_'
)

# ------------------------------------------------------
# 10. IMPORT INTO SAS MODEL MANAGER
# ------------------------------------------------------

pzmm.ImportModel.import_model(
    model_files=export_path,
    model_prefix=model_prefix,
    project="HDFC_POC_Project",
    input_data=X,
    predict_method=[best_model.predict_proba, [float, float]],
    score_metrics=["EM_CLASSIFICATION", "EM_EVENTPROBABILITY"],
    overwrite_model=True,
    target_values=["0", "1"],
    target_index=1,
    model_file_name=model_prefix + ".pickle",
    missing_values=True
)

print("\nModel successfully published to SAS Model Manager.")
