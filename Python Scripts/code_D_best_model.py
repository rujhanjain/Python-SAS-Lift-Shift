# ======================================================
# code_D_best_model.py
# Purpose: Train, evaluate, and register the champion model
# ======================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier   # Champion model
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Load training data
# ------------------------------------------------------
df = pd.read_csv("../Data/HDFC_TRAIN_PROCESSED.csv")
target = 'Response'
X = df.drop(columns=[target])
y = df[target]

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------
# 2. Define champion model
# ------------------------------------------------------
# Replace with your selected model if not Random
model_name = "Random"
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Preprocessing
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', 'passthrough', cat_cols)
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', model)
])

# ------------------------------------------------------
# 3. Train and predict
# ------------------------------------------------------
pipe.fit(X_train, y_train)
pred_proba = pipe.predict_proba(X_val)[:, 1]
pred_class = (pred_proba >= 0.5).astype(int)

# ------------------------------------------------------
# 4. Evaluate
# ------------------------------------------------------
def evaluate(y_true, y_pred_proba, y_pred_class):
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class, zero_division=0)
    rec = recall_score(y_true, y_pred_class, zero_division=0)
    f1 = f1_score(y_true, y_pred_class)
    
    df_eval = pd.DataFrame({'y': y_true, 'score': y_pred_proba}).sort_values('score', ascending=False)
    df_eval['cum_event'] = np.cumsum(df_eval['y']) / df_eval['y'].sum()
    df_eval['cum_non_event'] = np.cumsum(1 - df_eval['y']) / (len(df_eval) - df_eval['y'].sum())
    ks = max(abs(df_eval['cum_event'] - df_eval['cum_non_event']))
    lift = (df_eval.head(len(df_eval)//10)['y'].mean() / df_eval['y'].mean())
    
    metrics = {
        'Model': model_name,
        'AUC': round(auc, 4),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'KS': round(ks, 4),
        'Lift@Top10%': round(lift, 2)
    }
    return pd.DataFrame([metrics])

results = evaluate(y_val, pred_proba, pred_class)
print("\nChampion Model Evaluation:")
print(results.to_string(index=False))