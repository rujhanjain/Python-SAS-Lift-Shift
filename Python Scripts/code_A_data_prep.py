# ======================================================
# code_A_data_prep.py
# Purpose: Data preparation for HDFC Lift & Shift POC
# ======================================================

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
import os

os.makedirs('../Data', exist_ok=True)

# ------------------------------------------------------
# 1. Create SQLite database and load data
# ------------------------------------------------------
conn = sqlite3.connect('../pl_propensity.db')

def table_exists(conn, table_name):
    """Check if a table exists in SQLite database."""
    q = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    return pd.read_sql_query(q, conn).shape[0] > 0

for file, name in [
    ('../Data/DUMMY_PL_DATA_FOR_SAS_VIA_POC_1L_SAMPLE_TRAIN.csv', 'customer_pl_train'),
    ('../Data/DUMMY_PL_DATA_FOR_SAS_VIA_POC_1L_SAMPLE_OOT.csv', 'customer_pl_oot')
]:
    # If table exists, drop it for a clean reload
    if table_exists(conn, name):
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        print(f"Existing table '{name}' found and dropped for fresh load.")
    else:
        print(f"Table '{name}' not found — creating new.")

    # Load data in chunks for efficiency
    for chunk in pd.read_csv(file, chunksize=10000, low_memory=False):
        chunk.to_sql(name, conn, if_exists='append', index=False)
    print(f"Data loaded into SQLite table '{name}'.")

print("All data loaded successfully into SQLite.")

# ------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------
def run_query(q): return pd.read_sql_query(q, conn)
def run_modify(q): 
    with conn: conn.execute(q)

# ------------------------------------------------------
# 3. SQL-based diagnostics (PROC SQL equivalent)
# ------------------------------------------------------
cols = [c for c in run_query("PRAGMA table_info(customer_pl_train)")['name']
        if c not in ('DUMMY_ID', 'Response')]

# dynamic missingness check
miss_cases = [f"SUM(CASE WHEN [{c}] IS NULL THEN 1 ELSE 0 END) AS miss_{c}" for c in cols]
missing_sql = f"""
SELECT
    {', '.join(miss_cases)},
    COUNT(*) AS total_rows
FROM customer_pl_train;
"""

queries_sql = {
    "Target Distribution": """
        SELECT Response, COUNT(*) AS cnt,
               ROUND(COUNT(*) * 100.0 / total.tot, 2) AS pct
        FROM customer_pl_train
        CROSS JOIN (SELECT COUNT(*) AS tot FROM customer_pl_train) total
        GROUP BY Response ORDER BY Response;
    """,
    "Rows After Basic Filtering": """
        WITH filtered AS (
            SELECT * FROM customer_pl_train
            WHERE AGE BETWEEN 18 AND 80
              AND OFFER_AMT > 0
              AND FINAL_IRR BETWEEN 0 AND 50
        )
        SELECT COUNT(*) AS rows_kept,
               ROUND(100.0 * COUNT(*) / total.tot, 2) AS pct_kept
        FROM filtered
        CROSS JOIN (SELECT COUNT(*) AS tot FROM customer_pl_train) total;
    """,
    "Missing Values (all columns)": missing_sql,
    "Vintage Min-Max": """
        SELECT MIN(VINTAGE) AS min_v, MAX(VINTAGE) AS max_v,
               MIN(VINTAGE_DAYS) AS min_d, MAX(VINTAGE_DAYS) AS max_d
        FROM customer_pl_train;
    """,
    "Numeric Summary": """
        SELECT ROUND(AVG(OFFER_AMT),2) AS avg_offer,
               ROUND(AVG(FINAL_IRR),4) AS avg_irr,
               ROUND(AVG(FOIR),4) AS avg_foir,
               ROUND(AVG(FINAL_SALARY),2) AS avg_salary
        FROM customer_pl_train;
    """,
    "Segmentation NTB vs Existing": """
        SELECT NTB_TAG, COUNT(*) AS cnt,
               ROUND(100.0 * COUNT(*) / total.tot, 2) AS pct,
               ROUND(AVG(Response),6) AS response_rate
        FROM customer_pl_train
        CROSS JOIN (SELECT COUNT(*) AS tot FROM customer_pl_train) total
        GROUP BY NTB_TAG ORDER BY NTB_TAG;
    """,
    "High FOIR Customers": """
        SELECT COUNT(*) AS high_foir_cnt,
               ROUND(AVG(Response),6) AS response_rate_high_foir
        FROM customer_pl_train
        WHERE FOIR > 70;
    """
}

for name, sql in queries_sql.items():
    print(f"\n--- {name} ---")
    try:
        df = run_query(sql)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error in {name}: {e}")

print("\nSQL diagnostics complete.")

# ------------------------------------------------------
# 3b. Correlation check for VINTAGE vs VINTAGE_DAYS
# ------------------------------------------------------
sample_corr = run_query("""
    SELECT VINTAGE, VINTAGE_DAYS
    FROM customer_pl_train
    WHERE VINTAGE IS NOT NULL AND VINTAGE_DAYS IS NOT NULL
    LIMIT 30000
""")

if not sample_corr.empty:
    corr_val = sample_corr['VINTAGE'].corr(sample_corr['VINTAGE_DAYS'])
    print(f"\nCorrelation (VINTAGE vs VINTAGE_DAYS): {corr_val:.4f}")
    if corr_val > 0.90:
        print("High correlation detected; dropping redundant columns...")
        try:
            # Drop VINTAGE first
            run_modify("ALTER TABLE customer_pl_train DROP COLUMN VINTAGE;")
            run_modify("ALTER TABLE customer_pl_oot DROP COLUMN VINTAGE;")
        except Exception as e:
            print(f"Error dropping VINTAGE (may already be removed): {e}")
    
        try:
            # Drop DUMMY_ID next
            run_modify("ALTER TABLE customer_pl_train DROP COLUMN DUMMY_ID;")
            run_modify("ALTER TABLE customer_pl_oot DROP COLUMN DUMMY_ID;")
        except Exception as e:
            print(f"Error dropping DUMMY_ID (may already be removed): {e}")
    
        print("Dropped 'VINTAGE' and 'DUMMY_ID' successfully (if present).")
    else:
        print("No high correlation detected; retaining both columns.")
else:
    print("No valid rows for correlation check, dropping just DUMMY_ID")
    run_modify("ALTER TABLE customer_pl_train DROP COLUMN DUMMY_ID;")
    run_modify("ALTER TABLE customer_pl_oot DROP COLUMN DUMMY_ID;")

# ------------------------------------------------------
# 3c. New analytical SQL diagnostics to derive calculated columns
# ------------------------------------------------------
# These exploratory queries replicate PROC SQL analysis to justify engineered variables.
# They highlight income-to-debt behavior, FOIR vs salary patterns, and customer vintage risk tiers.

new_analytics_sql = {
    "Average FOIR by Salary Band": """
        SELECT 
            CASE 
                WHEN FINAL_SALARY < 30000 THEN '<30K'
                WHEN FINAL_SALARY BETWEEN 30000 AND 60000 THEN '30K-60K'
                WHEN FINAL_SALARY BETWEEN 60001 AND 120000 THEN '60K-120K'
                ELSE '>120K' 
            END AS salary_band,
            ROUND(AVG(FOIR),2) AS avg_foir,
            ROUND(AVG(Response),4) AS resp_rate,
            COUNT(*) AS cnt
        FROM customer_pl_train
        GROUP BY salary_band;
    """,
    "Debt vs Income vs Response": """
        SELECT 
            ROUND(AVG(TOTAL_EMI_AMT),2) AS avg_emi,
            ROUND(AVG(FINAL_SALARY),2) AS avg_income,
            ROUND(AVG(Response),4) AS avg_resp,
            ROUND(AVG(TOTAL_EMI_AMT / NULLIF(FINAL_SALARY,0)),4) AS avg_dti
        FROM customer_pl_train;
    """,
    "Vintage Bucketing Behavior": """
        SELECT 
            CASE 
                WHEN VINTAGE_DAYS < 180 THEN 'New'
                WHEN VINTAGE_DAYS BETWEEN 180 AND 720 THEN 'Established'
                ELSE 'Mature' 
            END AS vint_bucket,
            ROUND(AVG(Response),4) AS resp_rate,
            COUNT(*) AS cnt
        FROM customer_pl_train
        GROUP BY vint_bucket;
    """
}

for name, sql in new_analytics_sql.items():
    print(f"\n--- {name} ---")
    try:
        df = run_query(sql)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error in {name}: {e}")

print("\nExploratory SQL analysis complete — deriving calculated columns next...")

# ------------------------------------------------------
# 3d. Calculated columns creation based on insights
# ------------------------------------------------------
# Reasoning:
# 1. From 'Debt vs Income vs Response' — DTI ratio (debt_to_income_ratio) is a key predictor.
# 2. From 'Average FOIR by Salary Band' — we can categorize customers by income range (salary_band_flag).
# 3. From 'Vintage Bucketing Behavior' — customer maturity stage can indicate risk (vintage_bucket).

def apply_calculated_columns(conn, table_name):
    print(f"\nCreating calculated columns for table: {table_name}")

    queries_calc = [
        # Add DTI ratio
        f"ALTER TABLE {table_name} ADD COLUMN debt_to_income_ratio FLOAT;",
        f"""
        UPDATE {table_name}
        SET debt_to_income_ratio = 
            CASE 
                WHEN FINAL_SALARY > 0 THEN ROUND(TOTAL_EMI_AMT / FINAL_SALARY, 4)
                ELSE NULL 
            END;
        """,

        # Add income category
        f"ALTER TABLE {table_name} ADD COLUMN salary_band_flag TEXT;",
        f"""
        UPDATE {table_name}
        SET salary_band_flag =
            CASE 
                WHEN FINAL_SALARY < 30000 THEN 'LOW'
                WHEN FINAL_SALARY BETWEEN 30000 AND 60000 THEN 'MID'
                WHEN FINAL_SALARY BETWEEN 60001 AND 120000 THEN 'UPPER'
                ELSE 'HIGH'
            END;
        """,

        # Add customer vintage category
        f"ALTER TABLE {table_name} ADD COLUMN vintage_bucket TEXT;",
        f"""
        UPDATE {table_name}
        SET vintage_bucket =
            CASE 
                WHEN VINTAGE_DAYS < 180 THEN 'NEW'
                WHEN VINTAGE_DAYS BETWEEN 180 AND 720 THEN 'ESTABLISHED'
                ELSE 'MATURE'
            END;
        """
    ]

    for q in queries_calc:
        try:
            conn.execute(q)
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                print(f"Error executing query on {table_name}: {e}")

# Apply calculated columns to both train and OOT datasets
apply_calculated_columns(conn, 'customer_pl_train')
apply_calculated_columns(conn, 'customer_pl_oot')

for col in ['debt_to_income_ratio', 'salary_band_flag', 'vintage_bucket']:
    try:
        df_check = run_query(f"SELECT COUNT(DISTINCT {col}) as distinct_vals FROM customer_pl_train")
        print(f"{col}: {df_check['distinct_vals'][0]} distinct values")
    except:
        pass

print("\nCalculated columns successfully added and updated in both tables.")

# ------------------------------------------------------
# 4. Pull data to pandas for feature engineering
# ------------------------------------------------------
df_train = run_query("SELECT * FROM customer_pl_train")
df_train = df_train.convert_dtypes()

# Refresh list of columns from current DB state
cols_current = run_query("PRAGMA table_info(customer_pl_train)")['name'].tolist()
cols_current = [c for c in cols_current if c not in ('Response',)]

# ------------------------------------------------------
# 5. Analytical drops (zero variance / high missingness)
# ------------------------------------------------------
zero_var = df_train.columns[df_train.nunique() <= 1].tolist()

# Rebuild missingness SQL dynamically from current columns
miss_cases = [f"SUM(CASE WHEN [{c}] IS NULL THEN 1 ELSE 0 END) AS miss_{c}" for c in cols_current]
missing_sql_latest = f"""
SELECT
    {', '.join(miss_cases)},
    COUNT(*) AS total_rows
FROM customer_pl_train
"""

miss_raw = run_query(missing_sql_latest)
total_rows = miss_raw['total_rows'].iloc[0]

high_miss = []
for c in cols_current:
    miss_cnt = miss_raw[f'miss_{c}'].iloc[0]
    miss_pct = round(miss_cnt / total_rows * 100, 2)
    if miss_pct > 50:
        high_miss.append(c)

drop_cols = list(set(zero_var + high_miss))
df_train = df_train.drop(columns=drop_cols, errors='ignore')
print(f"\nDropped {len(drop_cols)} columns (zero variance / >50% missing).")

# ------------------------------------------------------
# 6. Correlation-based feature reduction
# ------------------------------------------------------
num_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
if len(num_cols) > 1:
    corr_matrix = df_train[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [j for i in upper.columns for j in upper.index if upper.loc[j, i] > 0.95]
    if to_drop_corr:
        print("Dropped due to high correlation:", list(set(to_drop_corr))[:10], "...")
        df_train = df_train.drop(columns=list(set(to_drop_corr)), errors='ignore')
        print(f"Removed {len(set(to_drop_corr))} highly correlated features.")
else:
    print("Not enough numeric columns for correlation pruning.")

# ------------------------------------------------------
# 7. Log-transform monetary variables
# ------------------------------------------------------
monetary_cols = [c for c in df_train.columns if any(k in c.upper() for k in ['AMT', 'SALARY', 'INVEST', 'EMI', 'CREDIT'])]
for col in monetary_cols:
    df_train[col] = df_train[col].clip(lower=0)
    df_train[f'log_{col}'] = np.log1p(df_train[col])
    df_train = df_train.drop(columns=[col])
print(f"Applied log1p to {len(monetary_cols)} monetary columns.")

# ------------------------------------------------------
# 8. Train/validation split
# ------------------------------------------------------
target = 'Response' if 'Response' in df_train.columns else None
X = df_train.drop(columns=[target]) if target else df_train
y = df_train[target] if target else None

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y is not None else None
)
val_df = pd.concat([X_val, y_val], axis=1)
print("Validation data split complete.")

# ------------------------------------------------------
# 9. Generate four validation samples of 25K each
# ------------------------------------------------------
if len(val_df) >= 100000:
    subsets = np.array_split(val_df.sample(frac=1, random_state=42).head(100000), 4)
else:
    subsets = np.array_split(val_df.sample(frac=1, random_state=42), 4)

names = ['HDFC_OOO_1_Q1', 'HDFC_OOO_2_Q2', 'HDFC_OOO_3_Q3', 'HDFC_OOO_4_Q4']
for df, name in zip(subsets, names):
    df.to_csv(f"../Data/{name}.csv", index=False)
    print(f"Saved validation subset: {name}")

# ------------------------------------------------------
# 10. Save processed training data
# ------------------------------------------------------
train_final = pd.concat([X_train, y_train], axis=1)
train_final.to_csv("../Data/HDFC_TRAIN_PROCESSED.csv", index=False)
print("Processed training data exported for model scoring phase.")

conn.close()
print("\nData preparation complete.")

if __name__ == "__main__":
    print("code_A_data_prep.py executed successfully.")