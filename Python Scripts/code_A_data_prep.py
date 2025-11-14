# ======================================================
# code_A_data_prep.py
# Purpose: Data preparation for HDFC Lift & Shift POC
# ======================================================

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
import os
import json

os.makedirs('../Data', exist_ok=True)
os.makedirs('../Models', exist_ok=True)

# ------------------------------------------------------
# Column shortening utilities (max 30 chars)
# ------------------------------------------------------
def shorten_col(name, max_len=30):
    """
    Shorten a single column name but keep it meaningful.
    Rules:
      - If <=30 chars, keep as is.
      - If contains log_, keep prefix log_ and shorten rest.
      - Otherwise, keep first 25 chars + hash suffix.
    """
    if len(name) <= max_len:
        return name

    if name.startswith("log_"):
        base = name[4:]
        return "log_" + base[:(max_len - 4)]

    return name[:max_len]

def apply_col_shortening(df):
    """Apply shortening to all dataframe columns."""
    new_cols = [shorten_col(c) for c in df.columns]
    df.columns = new_cols
    return df

# ------------------------------------------------------
# Helper Functions
# ------------------------------------------------------
def table_exists(conn, table_name):
    q = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    return pd.read_sql_query(q, conn).shape[0] > 0

def run_query(q, conn):
    return pd.read_sql_query(q, conn)

def run_modify(q, conn):
    with conn: conn.execute(q)

# ------------------------------------------------------
# 1. Load Data into SQLite
# ------------------------------------------------------
def create_or_replace_tables(conn):
    for file, name in [
        ('../Data/DUMMY_PL_DATA_FOR_SAS_VIA_POC_1L_SAMPLE_TRAIN.csv', 'customer_pl_train'),
        ('../Data/DUMMY_PL_DATA_FOR_SAS_VIA_POC_1L_SAMPLE_OOT.csv', 'customer_pl_oot')
    ]:
        if table_exists(conn, name):
            conn.execute(f"DROP TABLE IF EXISTS {name}")
            print(f"Existing table '{name}' dropped for fresh load.")

        for chunk in pd.read_csv(file, chunksize=10000, low_memory=False):
            chunk.to_sql(name, conn, if_exists='append', index=False)

        print(f"Data loaded into SQLite table '{name}'")

    print("All data loaded successfully into SQLite.")

# ------------------------------------------------------
# 2. Diagnostics (read-only)
# ------------------------------------------------------
def run_sql_diagnostics(conn):
    print("\n=== Running SQL Diagnostics ===")

    cols = [c for c in run_query("PRAGMA table_info(customer_pl_train)", conn)['name']
            if c not in ('DUMMY_ID', 'Response')]

    miss_cases = [f"SUM(CASE WHEN [{c}] IS NULL THEN 1 ELSE 0 END) AS miss_{c}" for c in cols]
    missing_sql = f"SELECT {', '.join(miss_cases)}, COUNT(*) AS total_rows FROM customer_pl_train;"

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
        try:
            print(f"\n--- {name} ---")
            print(run_query(sql, conn).to_string(index=False))
        except Exception as e:
            print(f"Error in {name}: {e}")

# ------------------------------------------------------
# 3. Handle VINTAGE Correlation
# ------------------------------------------------------
def handle_vintage_correlation(conn):
    sample_corr = run_query("""
        SELECT VINTAGE, VINTAGE_DAYS
        FROM customer_pl_train
        WHERE VINTAGE IS NOT NULL AND VINTAGE_DAYS IS NOT NULL
        LIMIT 30000
    """, conn)

    if sample_corr.empty:
        print("No correlation check data; dropping DUMMY_ID")
        for tbl in ['customer_pl_train', 'customer_pl_oot']:
            try: run_modify(f"ALTER TABLE {tbl} DROP COLUMN DUMMY_ID;", conn)
            except: pass
        return

    corr_val = sample_corr['VINTAGE'].corr(sample_corr['VINTAGE_DAYS'])
    print(f"\nCorrelation (VINTAGE vs VINTAGE_DAYS): {corr_val:.4f}")

    if corr_val > 0.9:
        print("High correlation detected — dropping VINTAGE and DUMMY_ID")
        for col in ['VINTAGE', 'DUMMY_ID']:
            for tbl in ['customer_pl_train', 'customer_pl_oot']:
                try: run_modify(f"ALTER TABLE {tbl} DROP COLUMN {col};", conn)
                except: pass

# ------------------------------------------------------
# 4. Calculated Columns
# ------------------------------------------------------
def apply_calculated_columns(conn, table_name):
    queries_calc = [
        f"ALTER TABLE {table_name} ADD COLUMN debt_to_income_ratio FLOAT;",
        f"""
        UPDATE {table_name}
        SET debt_to_income_ratio = 
            CASE 
                WHEN FINAL_SALARY > 0 THEN ROUND(TOTAL_EMI_AMT / FINAL_SALARY, 4)
                ELSE -1 
            END;
        """,
        f"ALTER TABLE {table_name} ADD COLUMN no_salary_flag INT;",
        f"UPDATE {table_name} SET no_salary_flag = CASE WHEN FINAL_SALARY <= 0 THEN 1 ELSE 0 END;",
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

# ------------------------------------------------------
# 5. TRAIN Feature Engineering
# ------------------------------------------------------
def feature_engineering_train(conn):
    df = run_query("SELECT * FROM customer_pl_train", conn)
    df = df.convert_dtypes()

    # SHORTEN COLUMNS FIRST
    df = apply_col_shortening(df)

    # Convert numeric-like
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Zero variance
    zero_var = df.columns[df.nunique(dropna=True) <= 1].tolist()

    # High missingness
    high_miss = [c for c in df.columns if df[c].isna().mean() * 100 > 50]

    drop_cols = list(set(zero_var + high_miss))

    # Correlation pruning
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    to_drop_corr = []
    if len(num_cols) > 1:
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]

    drop_cols = list(set(drop_cols + to_drop_corr))
    df = df.drop(columns=drop_cols, errors='ignore')

    # Monetary detection
    monetary_cols = [
        c for c in df.columns
        if any(k in c.upper() for k in ['AMT', 'SALARY', 'INVEST', 'EMI', 'CREDIT'])
    ]

    monetary_cols = [
        c for c in monetary_cols
        if df[c].nunique() > 5 and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in monetary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col].clip(lower=0)

        new_col = shorten_col(f"log_{col}")
        df[new_col] = np.log1p(df[col])

        df = df.drop(columns=[col], errors='ignore')

    # Shorten again after new columns created
    df = apply_col_shortening(df)

    # Save static filters
    static_filters = {
        "zero_var_drop": zero_var,
        "high_miss_drop": high_miss,
        "corr_drop": to_drop_corr,
        "final_keep": df.columns.tolist()
    }
    with open("../Models/static_filters.json", "w") as f:
        json.dump(static_filters, f, indent=2)

    return df

# ------------------------------------------------------
# 6. OOT Feature Engineering
# ------------------------------------------------------
def feature_engineering_oot(conn):
    df = run_query("SELECT * FROM customer_pl_oot", conn)
    df = df.convert_dtypes()

    # SHORTEN FIRST
    df = apply_col_shortening(df)

    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Identify monetary cols
    monetary_cols = [
        c for c in df.columns
        if any(k in c.upper() for k in ['AMT','SALARY','INVEST','EMI','CREDIT'])
    ]
    monetary_cols = [
        c for c in monetary_cols
        if df[c].nunique() > 5 and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in monetary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)

        new_col = shorten_col(f"log_{col}")
        df[new_col] = np.log1p(df[col])

        df = df.drop(columns=[col], errors='ignore')

    # Shorten again after log columns
    df = apply_col_shortening(df)

    # Align with train
    with open("../Models/static_filters.json") as f:
        filters = json.load(f)

    final_keep = filters["final_keep"]
    df = df.reindex(columns=final_keep, fill_value=0)

    return df

# ------------------------------------------------------
# 7. Main Orchestration
# ------------------------------------------------------
def main():
    conn = sqlite3.connect('../pl_propensity.db')

    create_or_replace_tables(conn)
    run_sql_diagnostics(conn)
    handle_vintage_correlation(conn)

    for tbl in ['customer_pl_train', 'customer_pl_oot']:
        apply_calculated_columns(conn, tbl)

    df_train = feature_engineering_train(conn)
    df_train = apply_col_shortening(df_train)
    df_train.to_csv("../Data/HDFC_TRAIN_PROCESSED.csv", index=False)

    feature_schema = df_train.columns.tolist()
    with open("../Models/feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)
    print(f"Saved feature schema: {len(feature_schema)} features.")

    df_oot = feature_engineering_oot(conn)
    df_oot.to_csv("../Data/HDFC_OOT_PROCESSED.csv", index=False)

    conn.close()
    print("\nData preparation complete.")

if __name__ == "__main__":
    main()
