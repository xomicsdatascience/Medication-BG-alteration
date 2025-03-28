import pandas as pd
import numpy as np
import pickle
import datetime as dt
import time
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency  
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

print('12 HOUR LOOKBACK')
demo = pd.read_csv('meyer_2306_encounters_major_class_2023_05_30.csv')
SH = pd.read_excel('Patient Info.xlsx')
PMH = pd.read_csv('Meyer2306 Diagnosis Hx.csv') 

# Read in meds input previously made
with open(r'final_meds_12H.pkl', 'rb') as handle:
    final_med_df = pickle.load(handle)
 
# Read in labs input previously made
with open(r'final_labs_12H.pkl', 'rb') as handle:
    final_lab_df = pickle.load(handle)
    
def prepare_static_inputs(demo, SH):
    demo_MRN = demo[['MRN', 'CSN', 'CONTACT_DATE', 'BMI', 'DPT_NAME']].drop_duplicates()
    static_inputs = demo_MRN.merge(SH, on='MRN', how='inner')
    # Select and drop duplicate columns
    static_inputs = static_inputs[['CSN', 'MRN', 'CONTACT_DATE', 'BMI', 'BIRTH_DATE', 'ETHNIC_GROUP',
                                   'SEX', 'RACE_1', 'SMOKE_TOB_LAST_STATUS', 'ILL_DRUG_LAST_STATUS',
                                   'ALCOHOL_LAST_STATUS', 'DPT_NAME']].drop_duplicates()
    # Calculate age in years
    static_inputs['AGE'] = ((pd.to_datetime(static_inputs['CONTACT_DATE']) - pd.to_datetime(static_inputs['BIRTH_DATE'])) 
                            / np.timedelta64(1, 'D')) / 365
    static_inputs = static_inputs[['MRN', 'CSN', 'AGE', 'BMI', 'ETHNIC_GROUP', 'SEX', 'RACE_1',
                                   'SMOKE_TOB_LAST_STATUS', 'ILL_DRUG_LAST_STATUS', 'ALCOHOL_LAST_STATUS',
                                   'DPT_NAME']]
    static_inputs = static_inputs.dropna(subset=['BMI'])
    static_inputs['MRN'] = static_inputs['MRN'].round().astype('Int64')
    return static_inputs
    
def extract_admit_diag(demo):
    admit_diag = demo[['MRN', 'CSN', 'ADMIT_DIAG']]
    return admit_diag
    
def classify_admission_diagnoses(admit_diag):
    # Define diagnosis patterns and corresponding column names
    diagnosis_dict = {
        'Admit_CHF': [
            'Acute on chronic congestive heart failure, unspecified heart failure type (HCC)',
            'Systolic CHF, acute on chronic (HCC)',
            'Congestive heart failure, unspecified HF chronicity, unspecified heart failure type (HCC)',
            'CHF',
            'Acute congestive heart failure, unspecified heart failure type (HCC)'],
        'Admit_Sepsis': [
            'Sepsis, due to unspecified organism',
            'Sepsis, due to unspecified organism, unspecified whether acute organ dysfunction present (HCC)',
            'Sepsis',
            'Severe sepsis (HCC)',
            'Septic shock (HCC)'],
        'Admit_GIB': [
            'gastrointestinal bleed',
            'GI bleed',
            'UGIB',
            'Gastrointestinal hemorrhage, unspecified gastrointestinal hemorrhage type'],
        'Admit_NV': [
            'Nausea and vomiting, intractability of vomiting not specified, unspecified vomiting type',
            'Non-intractable vomiting with nausea, unspecified vomiting type',
            'Intractable vomiting with nausea, unspecified vomiting type'],
        'Admit_AMS': [
            'Altered mental status, unspecified altered mental status type'],
        'Admit_AKI': [
            'Acute kidney injury (HCC)',
            'Acute renal failure, unspecified acute renal failure type (HCC)',
            'Renal insufficiency',
            'Acute renal insufficiency'],
        'Admit_ESRD': [
            'ESRD on dialysis (HCC)',
            'Chronic kidney disease, unspecified CKD stage',
            'ESRD',
            'ESRD on hemodialysis (HCC)']}
    
    # Iterate over diagnosis patterns and classify each diagnosis
    for column, patterns in diagnosis_dict.items():
        pattern = '|'.join(patterns)
        admit_diag[column] = np.where(admit_diag['ADMIT_DIAG'].str.contains(pattern, case=False, na=False), 1, 0)
    admit_diag['Admit_Pain'] = np.where(admit_diag['ADMIT_DIAG'].str.contains('pain', case=False, na=False), 1, 0)
    admit_diag.drop(['ADMIT_DIAG', 'MRN'], axis=1, inplace=True)
    admit_diag = admit_diag.groupby('CSN').sum()
    admit_diag[admit_diag > 1] = 1  
    admit_diag.reset_index(inplace=True)
    return admit_diag
    
def update_pmh_with_icd10(pmh_df):
    # Define ICD-10 patterns and corresponding column names
    icd10_dict = {
        'PMH_Liver_failure': 'K72',
        'PMH_CKD': 'N18',
        'PMH_T1DM': 'E10',
        'PMH_T2DM': 'E11',
        'PMH_CHF': 'I50',
        'PMH_HypoThyroid': 'E03.0',
        'PMH_HyperThyroid': 'E05',
        'PMH_Pregnancy': 'Z33.1',
        'PMH_Anemia': 'D64.9',
        'PMH_Hypoglycemia': 'E16.2',
        'PMH_Malignancy': 'C80.1',
        'PMH_Pain': 'R52'
    }
    for column, pattern in icd10_dict.items():
        pmh_df[column] = pmh_df['CURRENT_ICD10_LIST'].str.contains(pattern, na=False).astype(int)
    pmh_df = (pmh_df[['MRN'] + list(icd10_dict.keys())]
              .groupby('MRN')
              .sum()
              .clip(upper=1))
    return pmh_df

def categorize_dpt_name(dataframe, column_name):
    # Define categories
    or_values = [
        '6-ACU (6-PACU)', '8-ACU (8-PACU)', '3-ACU (3-PACU)', '7GI-ACU (GI LAB PACU)', '6-PACU', '7-PACU', '5-PACU', 
        '7-ACU (7-PACU)', 'GI LAB MAIN', '8-OR', '6-OR', '5-OR', 'CARDIAC CATH LAB MAIN', 'AHSP 5-PACU', '3-PACU', 
        'INTERVENTION RAD MAIN', '8-PRE-OP', '8-PACU', '7-OR', '3-PREOP', 'RT BRONCHOSCOPY MAIN', '5-ACU (5-PACU)',
        'GI LAB MAIN']
    non_icu_values = [
        '6-NE', '6-NW', '4-NW', '5-NW', '5-NE', '5-SE', '7-SE', '5-SW', '6-SE', '8-NE', '6-SW', '3-N', '7-NE', '7-NW', 
        '4-SE', '7-SW', '4-SW', '8-NW', '3N-UNIV', '8-SE', '8-SW', '3S-UNIV', '3SPT', 'MDRH 1 SOUTH', 'MDRH 1 NORTH', 
        '3-SW', '3-SE', '3-SW OB', '3-NW', '7 REHAB', 'MDRH MED SSU', '4S-MON', 'MDRH 1 EAST', '3-N MFCU', '3-NE', 
        '4-NE', 'OLD PEDS 4-NE', 'AHSP 5-ACU', '3-SE OB', 'AHSP 5-OSU', 'MDRH EMERGENCY DEPT', '2-SCCT PEDS', 
        '3-LDR', 'DIALYSIS ACUTE MAIN']
    icu_values = [
        '6-ICU', '8S-NSICU', '8N-ICU', '7S-RICU', '7N-MICU', '4N-CICU', '5S-SICU', '5N-SICU', '5N-SICU', '6S-CSICU', 
        '6N-CSICU', '4S-ICU', 'MDRH INTENSIVE CARE', '4S-PICU']
    
    # Perform replacements
    dataframe[column_name] = dataframe[column_name].replace(non_icu_values, 'non-ICU')
    dataframe[column_name] = dataframe[column_name].replace(icu_values, 'ICU')
    dataframe[column_name] = dataframe[column_name].replace(or_values, 'OR')
    return dataframe
    
static_inputs = prepare_static_inputs(demo, SH)
static_inputs = categorize_dpt_name(static_inputs, 'DPT_NAME')
admit_diag = extract_admit_diag(demo)
admit_diag = classify_admission_diagnoses(admit_diag)
static_inputs=pd.merge(admit_diag, static_inputs, on='CSN', how='inner')
PMH = update_pmh_with_icd10(PMH)
static_inputs = pd.merge(PMH, static_inputs, on='MRN', how='inner')
print(final_lab_df.columns)

labs_output = final_lab_df[['CSN', 'BG_RESULT_TIME', 'BG_VALUE','GLUCOSE-POC','AST (SGOT)', 'CREATININE', 'L-LACTATE', 'POTASSIUM', 'ALBUMIN']]
print(labs_output.columns)

# Merge all inputs/output
lab_static_output = static_inputs.merge(labs_output, on='CSN', how='inner')
print(lab_static_output.columns)
med_lab_static_output = lab_static_output.merge(final_med_df, on=['CSN', 'BG_RESULT_TIME', 'BG_VALUE'], how='inner')
print(med_lab_static_output.columns)    

missing_data = med_lab_static_output.isnull().sum()  # Count missing values in each column
total_rows = len(med_lab_static_output)              # Total number of rows in the DataFrame
missing_percentage = (missing_data / total_rows) * 100  # Calculate percentage

# Combine the results into a summary DataFrame
missing_summary = pd.DataFrame({
    'Missing Values': missing_data,
    'Missing Percentage (%)': missing_percentage})

# Display the columns with missing data
missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
print(missing_summary)

categorical_columns = ['ETHNIC_GROUP', 'SEX', 'RACE_1', 
                       'SMOKE_TOB_LAST_STATUS', 'ILL_DRUG_LAST_STATUS', 'ALCOHOL_LAST_STATUS']
                      
# Check for missingness in categorical columns
for col in categorical_columns:
    # Create a contingency table to check for association
    contingency_table = pd.crosstab(med_lab_static_output[col].isnull(), med_lab_static_output['BG_VALUE'])
    # Perform a Chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"{col} missingness and ORD_VALUE: chi2-stat={chi2_stat:.2f}, p-value={p_value:.3f}")

#Impute variables that are MAR
categorical_columns = ['ETHNIC_GROUP','SEX']
mode_imputer = SimpleImputer(strategy='most_frequent')
med_lab_static_output[categorical_columns] = mode_imputer.fit_transform(med_lab_static_output[categorical_columns])
print(f"Missing values in categorical columns after mode imputation: {med_lab_static_output[categorical_columns].isna().sum()}")

# Keep missing as a column for variables that are not MAR
columns_to_replace = ['RACE_1','SMOKE_TOB_LAST_STATUS', 'ILL_DRUG_LAST_STATUS', 'ALCOHOL_LAST_STATUS']
med_lab_static_output[columns_to_replace] = med_lab_static_output[columns_to_replace].fillna('missing')

# Descriptive Stats - Continuous variables 
continuous_vars = ['AGE', 'BMI', 'CREATININE', 'POTASSIUM', 'AST (SGOT)', 'ALBUMIN', 'L-LACTATE', 'GLUCOSE-POC']

for var in med_lab_static_output[continuous_vars].columns:
    print(f"** {var} **")
    
    # Filter to exclude zero values
    non_zero_values = med_lab_static_output[var][med_lab_static_output[var] != 0]
    
    # Calculate statistics
    median = non_zero_values.median()
    Q1 = non_zero_values.quantile(0.25)  # 25th percentile
    Q3 = non_zero_values.quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range (IQR)
    
    # Print results
    print(f"Median (non-zero): {median}")
    print(f"Q1 (25th percentile, non-zero): {Q1}")
    print(f"Q3 (75th percentile, non-zero): {Q3}")
    print(f"IQR (Interquartile Range, non-zero): {IQR}")
    print('-' * 40)


# Descriptive stats - categorical columns
categorical_columns = [
    'ETHNIC_GROUP', 'SEX', 'RACE_1', 'SMOKE_TOB_LAST_STATUS', 'ILL_DRUG_LAST_STATUS', 
    'ALCOHOL_LAST_STATUS', 'DPT_NAME', 'Admit_AKI', 'Admit_AMS', 'Admit_CHF', 'Admit_ESRD',
    'Admit_GIB', 'Admit_NV', 'Admit_Pain', 'Admit_Sepsis', 'PMH_Anemia', 'PMH_CHF', 
    'PMH_CKD', 'PMH_HyperThyroid', 'PMH_HypoThyroid', 'PMH_Hypoglycemia', 'PMH_Liver_failure', 
    'PMH_Malignancy', 'PMH_Pain', 'PMH_Pregnancy', 'PMH_T1DM', 'PMH_T2DM']
table_data = {}
for col in categorical_columns:
    counts = med_lab_static_output[col].value_counts()
    percentages = med_lab_static_output[col].value_counts(normalize=True) * 100
    col_df = pd.DataFrame({
        'Count': counts,
        'Percentage': percentages})
    table_data[col] = col_df
for col, data in table_data.items():
    print(f"\n{col}:\n{data}\n")

# Perform one-hot encoding for the specified columns with binary representation (1/0)
columns_to_encode = ['ETHNIC_GROUP', 'SEX', 'RACE_1', 
                     'SMOKE_TOB_LAST_STATUS', 'ILL_DRUG_LAST_STATUS', 'ALCOHOL_LAST_STATUS', 'DPT_NAME']
med_lab_static_output = pd.get_dummies(
    med_lab_static_output, 
    columns=columns_to_encode, 
    drop_first=True)
    
# Replace special characters in column names with underscores
special_chars = [' ', '.', '-', '/', '(', ')', '%', ',', '+', '&', '=']
for char in special_chars:
    med_lab_static_output.columns = med_lab_static_output.columns.str.replace(char, '_')

# Save results
with open(r'med_lab_static_output_12h_imputed.pkl', 'wb') as handle:
    pickle.dump(med_lab_static_output, handle)
print('med_lab_static_output_12h_imputed shape', med_lab_static_output.shape)

