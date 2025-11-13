# ======================================================
# code_D_best_model.py
# Purpose: Score champion model on OOT or new data
#          and optionally register model in SAS Model Manager
# ======================================================

import os
import json
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime

from code_A_data_prep import (
    apply_calculated_columns,
    feature_engineering
)
from utils_model_reporting import (
    evaluate_model,
    generate_confusion_matrix_plot,
    generate_decile_table,
    save_metrics_to_excel
)

# Optional SAS registration flag
REGISTER_TO_SAS = True

# ------------------------------------------------------
# 1. Load Champion Model
# ------------------------------------------------------
MODEL_PATH = "../Models/champion_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Champion model not found. Run code_B_model_score.py first.")

pipe = joblib.load(MODEL_PATH)
print("Champion model loaded successfully.")

# ------------------------------------------------------
# 2. Load and Prepare Input Data
# ------------------------------------------------------
OOT_PATH = "../Data/DUMMY_PL_DATA_FOR_SAS_VIA_POC_1L_SAMPLE_OOT.csv"
DB_PATH = "../pl_propensity.db"

if not os.path.exists(OOT_PATH):
    raise FileNotFoundError("OOT CSV not found at ../Data/")

conn = sqlite3.connect(DB_PATH)
oot_df_raw = pd.read_csv(OOT_PATH)
oot_df_raw.to_sql("customer_pl_oot", conn, if_exists="replace", index=False)
print("OOT data loaded into SQLite for consistent processing.")

apply_calculated_columns(conn, "customer_pl_oot")
df_oot = feature_engineering(conn, mode="oot")
conn.close()
print("OOT feature transformations applied successfully.")

# ------------------------------------------------------
# 3. Align Features and Predict
# ------------------------------------------------------
target = "Response" if "Response" in df_oot.columns else None
X_oot = df_oot.drop(columns=[target], errors="ignore") if target else df_oot
y_oot = df_oot[target] if target else None

with open("../Models/feature_schema.json") as f:
    train_cols = json.load(f)

X_oot = X_oot.reindex(columns=train_cols, fill_value=0)

print(f"Scoring data aligned to {len(train_cols)} training features.")

y_pred_proba = pipe.predict_proba(X_oot)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)
print("Model scoring completed successfully.")

# ------------------------------------------------------
# 4. Evaluate (if Response present)
# ------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
score_output_path = f"../Data/HDFC_SCORED_{timestamp}.csv"

df_oot["predicted_proba"] = y_pred_proba
df_oot["predicted_class"] = y_pred_class
df_oot.to_csv(score_output_path, index=False)
print(f"Scored data saved to {score_output_path}")

if y_oot is not None:
    print("\n=== Evaluation on OOT Data ===")
    metrics, detailed_df = evaluate_model("OOT", "Champion_Model", y_oot, y_pred_proba)

    cm_path = f"../Outputs/confusion_matrix_{timestamp}.png"
    generate_confusion_matrix_plot(y_oot, y_pred_class, title="OOT Confusion Matrix", save_path=cm_path)

    decile_table = generate_decile_table(y_oot, y_pred_proba)
    excel_path = "../Outputs/Scoring_Report.xlsx"
    save_metrics_to_excel(metrics, decile_table, excel_path)

    print(f"Evaluation complete. Report saved at {excel_path}")
else:
    print("No Response column found — skipping evaluation (pure scoring mode).")

print("\n=== Scoring Completed ===")

# ------------------------------------------------------
# 5. Optional SAS Viya Model Registration
# ------------------------------------------------------
if REGISTER_TO_SAS:
    try:
        import swat
        print("\nAttempting SAS Viya model registration...")

        conn_sas = swat.CAS("your-viya-host", 5570, "your_user", "your_pass")
        conn_sas.loadactionset("modelPublishing")

        conn_sas.modelPublishing.publishModel(
            modelTable={"name": "champion_model"},
            name="HDFC_Champion_Model",
            projectName="HDFC_Model_Project",
            repositoryName="GitHubRepo",
            replace=True
        )

        conn_sas.terminate()
        print("Model successfully registered to SAS Model Manager.")

    except Exception as e:
        print(f"SAS registration skipped: {e}")
else:
    print("SAS registration disabled (REGISTER_TO_SAS = False).")
