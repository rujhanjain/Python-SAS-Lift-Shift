# ======================================================
# utils_model_reporting.py
# Purpose: Save model evaluation metrics, confusion matrix,
#          decile analysis and metadata in Excel + image format
# ======================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# ------------------------------------------------------
# Utility: create report folder
# ------------------------------------------------------
def create_run_folder(base_dir="../Models/ModelReports"):
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