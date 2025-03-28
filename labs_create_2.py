import time
import pandas as pd
import numpy as np
import random
import bisect
import pickle
import matplotlib.pyplot as plt  
from datetime import timedelta


originalLabDf = pd.read_csv("meyer_2306_labs_with_order_time_2023_03_06.csv",encoding='windows-1252')

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

# Prepare lab data
inputLabDf = originalLabDf[['PAT_ENC_CSN_ID', 'RESULT_TIME', 'ORD_VALUE', 'COMPONENT_NAME']]
inputLabDf[['RESULT_TIME']] = inputLabDf[['RESULT_TIME']].apply(pd.to_datetime)
inputLabDf.rename(columns={'PAT_ENC_CSN_ID': 'CSN'}, inplace=True)
inputLabDf['ORD_VALUE'] = pd.to_numeric(inputLabDf['ORD_VALUE'], errors='coerce')
inputLabDf = inputLabDf.dropna(subset=['ORD_VALUE'])

# Clean duplicate lab names

replace_dict = {
    'GLUCOSE (POC)': 'GLUCOSE-POC',
    'SERUM CREATININE': 'CREATININE',
    'SERUM ALBUMIN': 'ALBUMIN'}
inputLabDf = inputLabDf.replace(replace_dict)
# Extract specific labs
lab_list = ['GLUCOSE-POC', 'CREATININE', 'POTASSIUM', 'AST (SGOT)', 'ALBUMIN', 'L-LACTATE']
targLabDf = inputLabDf[inputLabDf['COMPONENT_NAME'].isin(lab_list)]

def compute_lab_means_optimized(BGDf, targLabDf, time_window):
    """
    Compute mean lab values within a time window before BG_RESULT_TIME for each CSN and BG result.
    
    Parameters:
    - BGDf: DataFrame containing BG results with 'CSN', 'BG_RESULT_TIME', and 'BG_VALUE'.
    - targLabDf: DataFrame containing lab results with 'CSN', 'RESULT_TIME', 'COMPONENT_NAME', and 'ORD_VALUE'.
    - time_window: String specifying the time window, e.g., '12H', '24H'.
    
    Returns:
    - DataFrame with mean lab values per BG_RESULT_TIME, grouped by CSN, BG_RESULT_TIME, BG_VALUE, and COMPONENT_NAME.
    """
    # Convert time window string to timedelta
    time_delta = timedelta(hours=int(time_window[:-1]))

    # Convert timestamps to datetime
    BGDf['BG_RESULT_TIME'] = pd.to_datetime(BGDf['BG_RESULT_TIME'], errors='coerce')
    targLabDf['RESULT_TIME'] = pd.to_datetime(targLabDf['RESULT_TIME'], errors='coerce')

    # Pre-filter lab results to reduce computation
    min_bg_time = BGDf['BG_RESULT_TIME'].min()
    max_bg_time = BGDf['BG_RESULT_TIME'].max()
    filtered_labs = targLabDf[
        (targLabDf['RESULT_TIME'] >= min_bg_time - time_delta) &
        (targLabDf['RESULT_TIME'] <= max_bg_time)
    ]

    # Perform merge using a Cartesian join approach
    merged = pd.merge(
        BGDf, filtered_labs,
        on='CSN',
        suffixes=('_bg', '_lab')
    )

    # Filter rows within the time window
    merged = merged[
        (merged['RESULT_TIME'] >= merged['BG_RESULT_TIME'] - time_delta) &
        (merged['RESULT_TIME'] < merged['BG_RESULT_TIME'])
    ]

    # Group by relevant columns and compute mean lab values
    lab_means = (
        merged.groupby(['CSN', 'BG_RESULT_TIME', 'BG_VALUE', 'COMPONENT_NAME'])['ORD_VALUE']
        .mean()
        .unstack(fill_value=0)
        .reset_index()
    )

    return lab_means


# Generate DataFrames for 1H, 4H, and 12H windows
final_1H_df = compute_lab_means_optimized(BGDf, targLabDf, '1H')
with open(r'final_labs_1H.pkl', 'wb') as handle:
    pickle.dump(final_1H_df, handle)
print('final_1H_df shape:', final_1H_df.shape)

final_4H_df = compute_lab_means_optimized(BGDf, targLabDf, '4H')
with open(r'final_labs_4H.pkl', 'wb') as handle:
    pickle.dump(final_4H_df, handle)
print('final_4H_df shape:', final_4H_df.shape)

final_12H_df = compute_lab_means_optimized(BGDf, targLabDf, '12H')
with open(r'final_labs_12H.pkl', 'wb') as handle:
    pickle.dump(final_12H_df, handle)
print('final_12H_df shape:', final_12H_df.shape)

