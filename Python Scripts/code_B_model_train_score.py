# ======================================================
# code_B_model_score.py
# Purpose: Model training, evaluation, and OOT scoring
# Enhanced with automated reporting (Excel + plots + metadata)
# ======================================================

import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------
# Utility: create report folder
# ------------------------------------------------------
def create_run_folder(base_dir="../ModelReports"):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# ------------------------------------------------------
# Utility: confusion matrix + save as image
# ------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Non-Response", "Response"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return cm

# ------------------------------------------------------
# Utility: decile analysis table
# ------------------------------------------------------
def decile_analysis(y_true, y_pred_proba, dataset_name="Validation"):
    df = pd.DataFrame({'Actual': y_true, 'Score': y_pred_proba})
    df['Decile'] = pd.qcut(df['Score'], 10, labels=False, duplicates='drop') + 1
    df['Decile'] = 11 - df['Decile']  # rank from top
    grouped = df.groupby('Decile').agg(
        Count=('Actual', 'count'),
        Responders=('Actual', 'sum'),
        Avg_Score=('Score', 'mean')
    ).reset_index()
    grouped['Response_Rate'] = grouped['Responders'] / grouped['Count']
    grouped['Cumulative_Responders'] = grouped['Responders'].cumsum()
    grouped['Cumulative_Rate'] = grouped['Cumulative_Responders'] / grouped['Responders'].sum()
    grouped['Lift'] = grouped['Response_Rate'] / grouped['Response_Rate'].mean()
    grouped['Dataset'] = dataset_name
    return grouped

# ------------------------------------------------------
# Utility: Excel report generator
# ------------------------------------------------------
def save_model_report(run_dir, results_val, results_oot, df_decile_val, df_decile_oot, metadata):
    excel_path = os.path.join(run_dir, "metrics_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_val.to_excel(writer, sheet_name="Validation_Summary", index=False)
        if results_oot is not None:
            results_oot.to_excel(writer, sheet_name="OOT_Summary", index=False)
        df_decile_val.to_excel(writer, sheet_name="Decile_Validation", index=False)
        if df_decile_oot is not None:
            df_decile_oot.to_excel(writer, sheet_name="Decile_OOT", index=False)
        meta_df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Value'])
        meta_df.to_excel(writer, sheet_name="Run_Metadata")
    print(f"Excel report saved at: {excel_path}")

    meta_path = os.path.join(run_dir, "metadata.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved at: {meta_path}")

# ------------------------------------------------------
# 1. Load data
# ------------------------------------------------------
df = pd.read_csv("../Data/HDFC_TRAIN_PROCESSED.csv")
oot_df = pd.read_csv("../Data/HDFC_OOT_PROCESSED.csv")

target = ['Response', 'salary_band_flag', 'vintage_bucket']  # Example target and special columns
X = df.drop(columns=target, axis=1, errors='ignore')
y = df[target[0]]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Training data ki quality: {X_train.isna().sum()[X_train.isna().sum() > 0]}')

# ------------------------------------------------------
# 2. Define models
# ------------------------------------------------------
models = {
    'Logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(eval_metric='auc', random_state=42, use_label_encoder=False),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

# ------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

logistic_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', 'passthrough', cat_cols)
])

# ------------------------------------------------------
# 4. Evaluation function
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
    lift = (df_eval.head(len(df_eval)//10)['y'].mean() / df_eval['y'].mean())

    results = {
        'Dataset': dataset_name,
        'Model': model_name,
        'AUC': round(auc, 4),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'KS': round(ks, 4),
        'Lift@Top10%': round(lift, 2)
    }

    return results, y_pred

# ------------------------------------------------------
# 5. Train and evaluate
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

    res, y_pred_class = evaluate_model("Validation", name, y_val, pred)
    results.append(res)

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)
print("\n=== Validation Metrics ===")
print(results_df.to_string(index=False))

# ------------------------------------------------------
# 6. Select and save champion
# ------------------------------------------------------
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
os.makedirs("../Models", exist_ok=True)
joblib.dump(best_model, "../Models/champion_model.pkl")
print(f"\nChampion model saved: {best_model_name}")

# ------------------------------------------------------
# 7. Evaluate on OOT
# ------------------------------------------------------
X_oot = oot_df.drop(columns=target, axis=1, errors='ignore')
y_oot = oot_df['Response']
preds_oot = best_model.predict_proba(X_oot)[:, 1]
oot_results, y_pred_oot = evaluate_model("OOT", best_model_name, y_oot, preds_oot)
results_oot_df = pd.DataFrame([oot_results])

# ------------------------------------------------------
# 8. Generate Reports
# ------------------------------------------------------
run_dir = create_run_folder()

# Save plots
cm_val_path = os.path.join(run_dir, "confusion_matrix_val.png")
cm_oot_path = os.path.join(run_dir, "confusion_matrix_oot.png")
save_confusion_matrix(y_val, (pred >= 0.5).astype(int), "Validation Data", cm_val_path)
save_confusion_matrix(y_oot, y_pred_oot, "OOT Data", cm_oot_path)

# Decile analysis
decile_val = decile_analysis(y_val, pred, "Validation")
decile_oot = decile_analysis(y_oot, preds_oot, "OOT")

# Metadata
metadata = {
    "Champion_Model": best_model_name,
    "Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Train_Shape": list(X_train.shape),
    "Validation_Shape": list(X_val.shape),
    "OOT_Shape": list(X_oot.shape),
    "Feature_Count": len(X_train.columns)
}

# Save Excel + JSON report
save_model_report(run_dir, results_df, results_oot_df, decile_val, decile_oot, metadata)

print(f"\nAll model artifacts and reports saved under: {run_dir}")
