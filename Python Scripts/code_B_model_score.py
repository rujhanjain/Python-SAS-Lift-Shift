# ======================================================
# code_B_model_score.py
# Purpose: Model training, evaluation, and OOT scoring
# ======================================================

import pandas as pd
import numpy as np
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
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Load data and split
# ------------------------------------------------------
df = pd.read_csv("../Data/HDFC_TRAIN_PROCESSED.csv")
target = 'Response'
X = df.drop(columns=[target])
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
# 3. Preprocessing (only for Logistic)
# ------------------------------------------------------
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

logistic_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', 'passthrough', cat_cols)
])

# ------------------------------------------------------
# 4. Utility functions
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

def show_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix — {title}")
    print(pd.DataFrame(cm,
                       index=["Actual 0", "Actual 1"],
                       columns=["Pred 0", "Pred 1"]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Non-Response", "Response"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.show()

def plot_auc_bar(results_df):
    plt.figure(figsize=(8,4))
    plt.bar(results_df['Model'], results_df['AUC'], color='steelblue')
    plt.title("Model AUC Comparison (Validation Data)")
    plt.ylabel("AUC Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 5. Train and evaluate all models on validation set
# ------------------------------------------------------
results = []
trained_models = {}

for name, model in models.items():
    if name == 'Logistic':
        pipe = Pipeline([('prep', logistic_preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict_proba(X_val)[:, 1]
        trained_models[name] = pipe
    else:
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]
        trained_models[name] = model
    
    res, y_pred_class = evaluate_model("Validation", name, y_val, pred)
    results.append(res)

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)
print("\n=== Validation Metrics (From Train CSV split) ===")
print(results_df.to_string(index=False))

plot_auc_bar(results_df)

# ------------------------------------------------------
# 6. Select Champion Model
# ------------------------------------------------------
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\nChampion Model Selected: {best_model_name}")

# ------------------------------------------------------
# 7. Evaluate Champion on OOT
# ------------------------------------------------------
try:
    oot_df = pd.read_csv("../Data/DUMMY_PL_DATA_FOR_SAS_VIA_POC_1L_SAMPLE_OOT.csv")
    print("\nLoaded OOT data:", oot_df.shape)

    if 'Response' not in oot_df.columns:
        print("OOT data missing 'Response'; only scoring probabilities.")
        preds = best_model.predict_proba(
            oot_df.reindex(columns=X_train.columns, fill_value=0)
        )[:, 1]
        pd.DataFrame({'Predicted_Prob': preds}).to_csv("../Data/HDFC_OOT_SCORED.csv", index=False)
        print("OOT predictions saved to ../Data/HDFC_OOT_SCORED.csv")
    else:
        X_oot = oot_df.drop(columns=['Response'], errors='ignore')
        X_oot = X_oot.reindex(columns=X_train.columns, fill_value=0)
        y_oot = oot_df['Response']

        preds = best_model.predict_proba(X_oot)[:, 1]
        res_oot, y_pred_class = evaluate_model("OOT", best_model_name, y_oot, preds)
        results_oot_df = pd.DataFrame([res_oot])

        print("\n=== OOT Metrics (From OOT CSV) ===")
        print(results_oot_df.to_string(index=False))

        show_confusion_matrix(y_oot, y_pred_class, title="OOT Data")

except Exception as e:
    print(f"\nError during OOT evaluation: {e}")