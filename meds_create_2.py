import time 
import pandas as pd
import numpy as np
import random
import bisect
import pickle
import matplotlib.pyplot as plt  
from datetime import timedelta

originalLabDf = pd.read_csv("meyer_2306_labs_with_order_time_2023_03_06.csv",encoding='windows-1252')
originalMedDf = pd.read_csv("Meyer2306 All Meds.csv")

BGDf=originalLabDf[originalLabDf['PROC_NAME'].str.contains("POCT Glucose", na=False)]
BGDf[['RESULT_TIME']] = BGDf[['RESULT_TIME']].apply(pd.to_datetime)

# Replace various non-numeric values in the 'ORD_VALUE' column with numeric equivalents
BGDf['ORD_VALUE'].replace(['154bs', 'BS 175', 'bs120', '168 mg/dL', '159=bs', 'bs 120', '`185',
                            '250 mg/dL', 'BG 171 mg/dL', '250 mg/dL', '240 mg/dL', '130  mg/dL',
                            'BS 145', 'bs-134'],
                           ['154', '175', '120', '168', '159', '120', '185', '250', '171', '250', '240',
                            '130', '145', '134'], inplace=True)
BGDf['ORD_VALUE'] = pd.to_numeric(BGDf['ORD_VALUE'], errors='coerce')

# Drop rows where 'ORD_VALUE' has NaN values (i.e., non-numeric or missing)
BGDf = BGDf.dropna(subset=['ORD_VALUE'])
BGDf=BGDf[['PAT_ENC_CSN_ID', 'RESULT_TIME', 'ORD_VALUE']]
BGDf.rename(columns={'PAT_ENC_CSN_ID': 'CSN'}, inplace=True)
BGDf.rename(columns={'RESULT_TIME': 'BG_RESULT_TIME','ORD_VALUE':'BG_VALUE'}, inplace=True)

# Process Medication DataFrame
MedDf = originalMedDf[['MRN', 'CSN', 'MAR_TAKEN_TIME', 'MEDICATION_NM', 'MAR_SIG']]
MedDf = MedDf.dropna()
MedDf = MedDf[originalMedDf['MAR_SIG'] > 0].reset_index(drop=True)

# Define medication exclusion lists
flush_list=['SODIUM CHLORIDE 0.9 % (FLUSH) IJ SYRG',
       '0.9% NACL IV LINE FLUSH BAG',
       'HEPARIN SOLUTION 2 UNITS/ML FOR FLUSH INTRAPROCEDURE (CATH LAB ONLY)',
       'HEPARIN, PORCINE (PF) 1000 UNIT/ML IJ SOLN (ADULT FLUSH)',
       'HEPARIN (PORCINE) IN NACL (PF) 1000 UNITS/500ML FOR ECMO FLUSH IV SOLP',
       'HEPARIN (PORCINE) IN NACL (PF) 1000 UNITS/500ML FOR ECMO FLUSH IV SOLP BOLUS FROM BAG',
       'ALBUMIN 5% IN LR FOR PORTAL FLUSH MIXTURE (INTRAPROCEDURE)',
       'HEPARIN FLUSH INFUSION FOR THROMBOLYSIS (5 UNITS/ML) IV SOLN',
       'EPINEPHRINE IN STERILE WATER (EPI FLUSH) INTRAPROCEDURE ',
       'HEPARIN PERIOTONEAL CATHETER FLUSH PEDIATRIC',
       'WASH SOLUTION (COVID-19 STUDY DRUG) 5% HUMAN SERUM ALBUMIN FOR IV LINE FLUSH',
       'SODIUM CHLORIDE 0.9% FOR ECMO FLUSH IV SOL BOLUS FROM BAG',
       'HEPARIN (PORCINE) IN NACL 1000 UNITS/500ML NS FOR  CONGENITAL HEART ECMO FLUSH FROM BAG',
       '0.9% NACL (EG-01-1962-03 STUDY DRUG) INTRAVENTRICULAR FLUSH- IRB #44821',
       'HEPARIN (PORCINE) IN NACL 500 UNITS/500ML NS FOR  CONGENITAL HEART ECMO FLUSH FROM BAG',
       'SODIUM CHLORIDE 0.9% FOR ECMO FLUSH',
       'ALBUMIN 25% IN LR FOR PORTAL FLUSH MIXTURE (INTRAPROCEDURE)',
       'HEPARIN LOCK FLUSH (PORCINE)  10 UNITS/ML IV SOLN',
       'SODIUM CHLORIDE 0.9 % IV SOLP']
irrig_list=['NEOMYCIN-POLYMYXIN-BACITRACIN IRRIGATION',
       'WATER FOR IRRIGATION, STERILE IR SOLN',
       'CEFAZOLIN (ANCEF)-VANCO-GENT-CLINDAMYCIN-BACITRACIN-TOBRA IRRIGATION MIXTURE (INTRAPROCEDURE)',
       'LACTATED RINGERS IRRIGATION INTRAPROCEDURE',
       'PAPAVERINE IN NS FOR IRRIGATION (FOR CARDIAC) MIXTURE (INTRAPROCEDURE)',
       'SIMETHICONE IN STERILE WATER FOR IRRIGATION MIXTURE (INTRAPROCEDURE)',
       'BACITRACIN IN NS FOR IRRIGATION MIXTURE (INTRAPROCEDURE)',
       'VANCOMYCIN 500 MG IN 200 ML 0.9% NACL POCKET IRRIGATION FOR INTRAPROCEDURE',
       'GENTAMICIN 80 MG IN 200 ML 0.9% NACL POCKET IRRIGATION FOR INTRAPROCEDURE',
       'AMPHOTERICIN B IRRIGATION',
       'NEOMYCIN-POLYMYXIN B (NEOSPORIN) IRRIGATION SOLUTION MIXTURE (INTRAPROCEDURE)',
       'BETADINE 10%, NS IRRIGATION MIXTURE (INTRAPROCEDURE)',
       'POLYMYXIN B IRRIGATION SOLUTION (INTRAPROCEDURE)',
       'AMPHOTERICIN B (FUNGIZONE) IRRIGATION MIXTURE (INTRAPROCEDURE)',
       'PAPAVERINE, VERAPAMIL IN NS FOR IRRIGATION (FOR CARDIAC) INTRAPROCEDE',
       'EPINEPHRINE IN NORMAL SALINE FOR IRRIGATION MIXTURE (INTRAPROCEDURE)',
       'GENTAMICIN IN NS IRRIGATION INTRAPROCEDURE ONLY',
       'NEOSPORIN-POLYMYXIN B IRRIGATION SOLUTION',
       'IRRIGATION SOLNS PHYSIOLOGICAL IR SOLN',
       'CARDIOPLEGIC SOLUTION WITH GLUTAMATE/ASPARTATE IRRIGATION',
       'ALUM 10 GRAM IN STERILE WATER 1,000 ML IRRIGATION',
       'BACTROBAN NASAL IRRIGATION',
       'BALANCED SALTS FOR INTRAOCULAR IRRIGATION MIXTURE (INTRAPROCEDURE)','SODIUM CHLORIDE  0.9 % IR SOLN']
iohexol=['IOHEXOL 300 MG IODINE/ML IV SOLN (RADIANT USE)','IOHEXOL 350 MG IODINE/ML IV SOLN']
NS_containing=['ARTIFICIAL SALIVA (YERBAS-LYT) MM SPRA',
 'ARTIFICIAL TEARS(HYPROMELLOSE) 0.5 % OP DROP', 'SODIUM CHLORIDE  0.65 % NA SPRA',
 'SODIUM CHLORIDE  NEB','ELECTROLYTE-A IV SOLP - BOLUS']
non_meds=['DRUG LEVEL REMINDER', 'HELP', 'IMS TEMPLATE', 'IMPELLA PURGE SOLUTION BUILDER', 'MMX ORAL SUSPENSION (IP ORDERING ONLY)']

# Filter out non-medications
MedDf = MedDf[~MedDf['MEDICATION_NM'].isin(irrig_list + flush_list + non_meds + iohexol + NS_containing)]

# Replace medication names as needed
replace_dict = {
    'INSULIN LISPRO 100 UNITS/ML SC SOLN': 'INSULIN LISPRO 100 UNITS/ML SOLN',
    'INSULIN LISPRO 100 UNITS/ML SC SOLN (SLIDING SCALE)': 'INSULIN LISPRO 100 UNITS/ML SOLN',
    'INSULIN REGULAR HUMAN 100 UNITS/ML IJ SOLN (SLIDING SCALE)': 'INSULIN REGULAR HUMAN 100 UNITS/ML IJ SOLN',
    'INSULIN REGULAR IN 0.9 % NACL 100 UNIT/100 ML (1 UNIT/ML) IV SOLN': 'INSULIN REGULAR IN 0.9% NACL 100 UNITS/100 ML (ADULT)',
    'INSULIN ASPART U-100 100 UNITS/ML SC SOLN (SLIDING SCALE)': 'INSULIN ASPART U-100 100 UNITS/ML SC SOLN',
    'INSULIN REGULAR HUMAN 100 UNIT/ML IJ SOLN (PRL 1772 & 2600  ONLY)': 'INSULIN REGULAR HUMAN 100 UNITS/ML IJ SOLN',
    'INSULIN GLARGINE 100 UNITS/ML SC SOLN': 'INSULIN GLARGINE 100 UNITS/ML',
    'INSULIN GLARGINE 100 UNITS/ML SC INPN': 'INSULIN GLARGINE 100 UNITS/ML',
    'INSULIN NPH AND REGULAR HUMAN 100 UNIT/ML (70-30) SC SUSP': 'INSULIN NPH AND REGULAR HUMAN 100 UNIT/ML (70-30) SC'
}
MedDf = MedDf.replace(replace_dict)

# Select target medications and top medications list
target_med_list=['INSULIN LISPRO 100 UNITS/ML SOLN',
                 'INSULIN REGULAR HUMAN 100 UNITS/ML IJ SOLN',
                 'INSULIN REGULAR IN 0.9% NACL 100 UNITS/100 ML (ADULT)',
                 'INSULIN ASPART U-100 100 UNITS/ML SC SOLN',
                 'INSULIN GLARGINE 100 UNITS/ML',
                 'INSULIN NPH AND REGULAR HUMAN 100 UNIT/ML (70-30) SC',
                 'SITAGLIPTIN 100 MG PO TABS', 'SITAGLIPTIN  50 MG PO TABS','SITAGLIPTIN  25 MG PO TABS',
                 'GLIMEPIRIDE 4 MG PO TABS', 'GLIMEPIRIDE 1 MG PO TABS','GLIMEPIRIDE 2 MG PO TABS',
                 'GLIPIZIDE 10 MG PO TABS', 'GLIPIZIDE  5 MG PO TABS',
                 'GLIPIZIDE  5 MG PO TR24','GLIPIZIDE 10 MG PO TR24',
                 'GLYBURIDE 1.25 MG PO TABS','GLYBURIDE 5 MG PO TABS',
                 'PIOGLITAZONE 30 MG PO TABS', 'PIOGLITAZONE 15 MG PO TABS','PIOGLITAZONE 45 MG PO TABS',
                 'CSHS PARENTERAL NUTRITITION ORDER ADULT','CSHS PERIPHERAL PARENTERAL NUTRITION ADULT','CSHS PARENTERAL NUTRITION ORDER CLINIMIX E MIXTURE']   
topNum = 300
top300_list = MedDf['MEDICATION_NM'].value_counts()[:topNum].index.to_list()
top300_DM_meds = target_med_list + top300_list
topMedDf = MedDf[MedDf['MEDICATION_NM'].isin(top300_DM_meds)]
topMedDf['MAR_TAKEN_TIME'] = pd.to_datetime(topMedDf['MAR_TAKEN_TIME'], errors='coerce')

# Step 1: BG Processing for BG_VALUE Summation within Time Window
BGDf['BG_RESULT_TIME'] = pd.to_datetime(BGDf['BG_RESULT_TIME'])
BGDf = BGDf.sort_values(by=['CSN', 'BG_RESULT_TIME'])

def sum_medication_doses_optimized(BGDf, med_df, time_window):
    time_delta = pd.Timedelta(time_window)

    # Convert relevant columns to datetime
    med_df['MAR_TAKEN_TIME'] = pd.to_datetime(med_df['MAR_TAKEN_TIME'], errors='coerce')
    BGDf['BG_RESULT_TIME'] = pd.to_datetime(BGDf['BG_RESULT_TIME'], errors='coerce')

    # Drop rows with invalid times
    med_df = med_df.dropna(subset=['MAR_TAKEN_TIME'])
    BGDf = BGDf.dropna(subset=['BG_RESULT_TIME'])

    # Merge BGDf with med_df on 'CSN' to filter medications for each BG result
    merged = pd.merge(
        BGDf[['CSN', 'BG_RESULT_TIME', 'BG_VALUE']],
        med_df[['CSN', 'MAR_TAKEN_TIME', 'MEDICATION_NM', 'MAR_SIG']],
        on='CSN'
    )

    # Filter medications within the specified time window
    merged = merged[
        (merged['MAR_TAKEN_TIME'] >= merged['BG_RESULT_TIME'] - time_delta) &
        (merged['MAR_TAKEN_TIME'] < merged['BG_RESULT_TIME'])
    ]

    # Group by 'CSN', 'BG_RESULT_TIME', 'BG_VALUE', and 'MEDICATION_NM' to sum doses
    dose_sums = (
        merged.groupby(['CSN', 'BG_RESULT_TIME', 'BG_VALUE', 'MEDICATION_NM'])['MAR_SIG']
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Return the resulting DataFrame
    return dose_sums

# Step 3: Get Medication Sums for Different Time Windows
# Summing medication doses for 1 hours
final_1H_df = sum_medication_doses_optimized(BGDf, topMedDf, '1H')

# Summing medication doses for 4 hours
final_4H_df = sum_medication_doses_optimized(BGDf, topMedDf, '4H')

# Summing medication doses for 12 hours
final_12H_df = sum_medication_doses_optimized(BGDf, topMedDf, '12H')

# Replace NaN values for medication columns with 0
final_1H_df = final_1H_df.fillna(0)
final_4H_df = final_4H_df.fillna(0)
final_12H_df = final_12H_df.fillna(0)

# Save results
with open(r'final_meds_1H.pkl', 'wb') as handle:
    pickle.dump(final_1H_df, handle)
with open(r'final_meds_4H.pkl', 'wb') as handle:
    pickle.dump(final_4H_df, handle)
with open(r'final_meds_12H.pkl', 'wb') as handle:
    pickle.dump(final_12H_df, handle)


