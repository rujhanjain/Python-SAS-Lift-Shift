/* ===============================================================
   code_A_data_prep_rewrite_oot_only.sas
   =============================================================== */

options validvarname=upcase;

/* ---------------------------------------------------------------
   1. Check existence of expected OOT source table
   --------------------------------------------------------------- */
proc sql;
    title "Check existence of expected OOT table in SPEEDY";
    select memname
    from dictionary.tables
    where libname="SPEEDY"
      and memname = "DUMMY_PL_DATA_OOT";
quit;
title;

/* ---------------------------------------------------------------
   2. Create HDFC_POC_OOT with calculated columns (visible SQL)
   --------------------------------------------------------------- */
/* 1. Build OOT with your flags + buckets exactly as written */
proc sql;
    create table SPEEDY.HDFC_POC_OOT as
    select 
        *,
        /* 1. No salary flag */
        (case when FINAL_SALARY <= 0 then 1 else 0 end) as no_salary_flag,
        
        /* 2. Debt-to-income ratio */
        (case when FINAL_SALARY > 0 then TOTAL_EMI_AMT / FINAL_SALARY 
            else -1 end) as debt_to_income_ratio,
        
        /* 3. Salary band flag */
        (case when FINAL_SALARY < 30000 then "LOW" 
              when FINAL_SALARY <= 60000 then "MID"
              when FINAL_SALARY <= 120000 then "UPPER" 
              else "HIGH" end) as salary_band_flag,
        
        /* 4. Vintage bucket */
        (case 
            when VINTAGE_DAYS < 180 then "NEW" 
            when VINTAGE_DAYS <= 720 then "ESTABLISHED"
            else "MATURE" end) as vintage_bucket

    from SPEEDY.DUMMY_PL_DATA_OOT (drop=VINTAGE);
quit;

/* ---------------------------------------------------------------
   3. Diagnostics (visible SQL + PROC MEANS)
   --------------------------------------------------------------- */

/* 3. Missing counts (selected columns) for OOT */
title "Data Summary";
proc means data=SPEEDY.DUMMY_PL_DATA_TRAIN nmiss n mean std min max;
run;
title;

data PYS3.HDFC_POC_OOT;
    set Speedy.HDFC_POC_OOT;
run;


/* Temp Build of Train for POC */
/* 1. Build OOT with your flags + buckets exactly as written */
proc sql;
    create table SPEEDY.HDFC_POC_TRAIN as
    select 
        *,
        /* 1. No salary flag */
        (case when FINAL_SALARY <= 0 then 1 else 0 end) as no_salary_flag,
        
        /* 2. Debt-to-income ratio */
        (case when FINAL_SALARY > 0 then TOTAL_EMI_AMT / FINAL_SALARY 
            else -1 end) as debt_to_income_ratio,
        
        /* 3. Salary band flag */
        (case when FINAL_SALARY < 30000 then "LOW" 
              when FINAL_SALARY <= 60000 then "MID"
              when FINAL_SALARY <= 120000 then "UPPER" 
              else "HIGH" end) as salary_band_flag,
        
        /* 4. Vintage bucket */
        (case 
            when VINTAGE_DAYS < 180 then "NEW" 
            when VINTAGE_DAYS <= 720 then "ESTABLISHED"
            else "MATURE" end) as vintage_bucket

    from SPEEDY.DUMMY_PL_DATA_TRAIN (drop=VINTAGE);
quit;

data PYS3.HDFC_POC_TRAIN;
    set Speedy.HDFC_POC_TRAIN;
run;