import sqlite3
import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------
# 0. Setup: define input/output paths
# -----------------------------------------------------------

RAW_OOT_CSV = "HDFC_POC_OOT.csv"
FINAL_OOT_CSV = "OUTPUT_OOT_FOR_SAS.csv"
SQLITE_DB = "local_dataprep.db"

# Clean old DB when re-running
if os.path.exists(SQLITE_DB):
    os.remove(SQLITE_DB)

# -----------------------------------------------------------
# 1. Load raw CSV into Pandas
# -----------------------------------------------------------

df = pd.read_csv(RAW_OOT_CSV)

# -----------------------------------------------------------
# 2. Create SQLite DB and load data
# -----------------------------------------------------------

conn = sqlite3.connect(SQLITE_DB)
df.to_sql("oot_raw", conn, index=False, if_exists="replace")

# -----------------------------------------------------------
# 3. Create OOT working table with calculated columns
# -----------------------------------------------------------

create_sql = """
CREATE TABLE oot_prepared AS
SELECT
    *,
    
    /* Example Calculated Columns */
    CASE WHEN AGE IS NULL THEN -1 ELSE AGE END AS AGE_IMPUTED,
    
    /* Example bucket */
    CASE 
        WHEN INCOME < 20000 THEN 'LOW'
        WHEN INCOME BETWEEN 20000 AND 50000 THEN 'MID'
        ELSE 'HIGH'
    END AS INCOME_BUCKET,
    
    /* Log transform with safe-guard */
    CASE 
        WHEN AMOUNT_DUE <= 0 OR AMOUNT_DUE IS NULL THEN 0
        ELSE LOG(AMOUNT_DUE)
    END AS LOG_AMOUNT_DUE

FROM oot_raw;
"""

conn.execute("DROP TABLE IF EXISTS oot_prepared;")
conn.execute(create_sql)
conn.commit()

# -----------------------------------------------------------
# 4. Diagnostics (NULL counts, summary stats, distribution checks)
# -----------------------------------------------------------

diagnostics = {}

# NULL counts
null_count_sql = """
SELECT 
    SUM(CASE WHEN AGE IS NULL THEN 1 ELSE 0 END) AS NULL_AGE,
    SUM(CASE WHEN INCOME IS NULL THEN 1 ELSE 0 END) AS NULL_INCOME,
    SUM(CASE WHEN AMOUNT_DUE IS NULL THEN 1 ELSE 0 END) AS NULL_AMOUNT_DUE
FROM oot_raw;
"""

diagnostics["null_counts"] = pd.read_sql(null_count_sql, conn)

# Summary stats
summary_sql = """
SELECT 
    AVG(AGE) AS AGE_MEAN,
    AVG(INCOME) AS INCOME_MEAN,
    AVG(AMOUNT_DUE) AS AMOUNT_DUE_MEAN
FROM oot_raw;
"""

diagnostics["summary_stats"] = pd.read_sql(summary_sql, conn)

# Distribution check on income bucket
bucket_sql = """
SELECT INCOME_BUCKET, COUNT(*) AS CNT
FROM oot_prepared
GROUP BY INCOME_BUCKET;
"""

diagnostics["bucket_distribution"] = pd.read_sql(bucket_sql, conn)

# Export diagnostics to console
print("\n===== DIAGNOSTICS =====")
for key, dfout in diagnostics.items():
    print(f"\n--- {key.upper()} ---")
    print(dfout)

# -----------------------------------------------------------
# 5. Export prepared OOT data for SAS S3 caslib (PYS3)
# -----------------------------------------------------------

oot_prepared_df = pd.read_sql("SELECT * FROM oot_prepared;", conn)
oot_prepared_df.to_csv(FINAL_OOT_CSV, index=False)

print("\nFinal prepared OOT CSV written:", FINAL_OOT_CSV)
conn.close()
