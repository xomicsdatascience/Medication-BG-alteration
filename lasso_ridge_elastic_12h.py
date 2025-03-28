import pandas as pd
import numpy as np
import pickle
import datetime as dt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_squared_error, r2_score, PredictionErrorDisplay
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import Lasso, LassoCV, Ridge, ElasticNet
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedKFold,cross_validate,GroupShuffleSplit
import scipy as sp 
import statsmodels.api
from matplotlib.ticker import FormatStrFormatter
from sklearn.experimental import enable_iterative_imputer  # Enables IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import time
import seaborn as sns
import re
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.cluster import KMeans
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Read in meds input previously made
with open(r'med_lab_static_output_12h_imputed.pkl', 'rb') as handle:
    med_lab_static_output = pickle.load(handle)
    
print('12 HOUR')

# Convert boolean columns to integers (vectorized)
med_lab_static_output = med_lab_static_output.convert_dtypes()  # Ensures proper dtypes for conversion
med_lab_static_output.loc[:, med_lab_static_output.dtypes == 'boolean'] = med_lab_static_output.loc[:, med_lab_static_output.dtypes == 'boolean'].astype(int)

# Prepare features (X) and target (y)
y = med_lab_static_output['BG_VALUE']
X = med_lab_static_output.drop(columns=['BG_VALUE','MRN', 'BG_RESULT_TIME'])
groups = med_lab_static_output['CSN']

# Create a GroupShuffleSplit instance
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Perform the split
for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Verify no overlap in CSNs
train_csns = set(X_train['CSN'])
test_csns = set(X_test['CSN'])
assert train_csns.isdisjoint(test_csns), "Train and test sets have overlapping CSNs"

# Number of folds for cross-validation
n_splits = 10

# Initialize lists to store coefficients from each fold
all_coefficients_lasso = []
all_coefficients_ridge = []
all_coefficients_en = []

# Define GroupKFold cross-validator
gkf = GroupKFold(n_splits=n_splits)

# Loop over the folds, ensuring no CSN overlap
for train_idx, test_idx in gkf.split(X_train, y_train, groups=X_train['CSN']):
    X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

    # Fit the Lasso model
    lasso_fold = Lasso(alpha=0.1)
    lasso_fold.fit(X_train_fold, y_train_fold)
    all_coefficients_lasso.append(lasso_fold.coef_)

    # Fit the Ridge model
    ridge_fold = Ridge(alpha=0.1)
    ridge_fold.fit(X_train_fold, y_train_fold)
    all_coefficients_ridge.append(ridge_fold.coef_)

    # Fit the ElasticNet model
    en_fold = ElasticNet(alpha=0.1, l1_ratio=0.5)
    en_fold.fit(X_train_fold, y_train_fold)
    all_coefficients_en.append(en_fold.coef_)

# Convert list of coefficients to numpy arrays
all_coefficients_lasso = np.array(all_coefficients_lasso)
all_coefficients_ridge = np.array(all_coefficients_ridge)
all_coefficients_en = np.array(all_coefficients_en)

# Calculate the average coefficients across all folds
avg_coefficients_lasso = np.mean(all_coefficients_lasso, axis=0)
avg_coefficients_ridge = np.mean(all_coefficients_ridge, axis=0)
avg_coefficients_en = np.mean(all_coefficients_en, axis=0)

# Final model fitting and evaluation
final_lasso = Lasso(alpha=0.1)
final_lasso.fit(X_train, y_train)
y_pred_lasso = final_lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

final_ridge = Ridge(alpha=0.1)
final_ridge.fit(X_train, y_train)
y_pred_ridge = final_ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

final_en = ElasticNet(alpha=0.1, l1_ratio=0.5)
final_en.fit(X_train, y_train)
y_pred_en = final_en.predict(X_test)
mse_en = mean_squared_error(y_test, y_pred_en)
rmse_en = np.sqrt(mse_en)
r2_en = r2_score(y_test, y_pred_en)

# Print results
print(f"Final Lasso Model - MSE: {mse_lasso:.4f}, RMSE: {rmse_lasso:.4f}, R2: {r2_lasso:.4f}")
print(f"Final Ridge Model - MSE: {mse_ridge:.4f}, RMSE: {rmse_ridge:.4f}, R2: {r2_ridge:.4f}")
print(f"Final ElasticNet Model - MSE: {mse_en:.4f}, RMSE: {rmse_en:.4f}, R2: {r2_en:.4f}")

# Store and print the average coefficients (optional)
avg_coefficients_lasso_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Average Coefficient': avg_coefficients_lasso
}).sort_values(by='Average Coefficient', ascending=False)

avg_coefficients_ridge_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Average Coefficient': avg_coefficients_ridge
}).sort_values(by='Average Coefficient', ascending=False)

avg_coefficients_en_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Average Coefficient': avg_coefficients_en
}).sort_values(by='Average Coefficient', ascending=False)

# Save the sorted average coefficients to a file
with open(f'avg_coefficients_lasso_12h_df.pkl', 'wb') as handle:
    pickle.dump(avg_coefficients_lasso_df, handle)

# Select non-zero coefficient features
def get_nonzero_features(avg_coefficients_df):
    nonzero_features = avg_coefficients_df[avg_coefficients_df['Average Coefficient'] != 0]['Feature']
    return nonzero_features

nonzero_features_lasso = get_nonzero_features(avg_coefficients_lasso_df)
nonzero_features_ridge = get_nonzero_features(avg_coefficients_ridge_df)
nonzero_features_en = get_nonzero_features(avg_coefficients_en_df)

num_non_zero_lasso = len(nonzero_features_lasso)
num_non_zero_ridge = len(nonzero_features_ridge)
num_non_zero_en = len(nonzero_features_en)

print(f"Number of non-zero average coefficients for Lasso: {num_non_zero_lasso}")
print(f"Number of non-zero average coefficients for Ridge: {num_non_zero_ridge}")
print(f"Number of non-zero average coefficients for ElasticNet: {num_non_zero_en}")

X = X.astype(float)

# Compute VIF for selected features
def compute_vif(X, nonzero_features):
    if len(nonzero_features) > 0:
        X_selected = X[nonzero_features]
        vif_data = pd.DataFrame({
            'Feature': X_selected.columns,
            'VIF': [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]
        })

        # Compute VIF category percentages
        num_vif_lt5 = (vif_data['VIF'] < 5).sum()
        num_vif_5to10 = ((vif_data['VIF'] >= 5) & (vif_data['VIF'] < 10)).sum()
        num_vif_gt10 = (vif_data['VIF'] >= 10).sum()

        total_vif = len(vif_data)
        pct_vif_lt5 = (num_vif_lt5 / total_vif) * 100
        pct_vif_5to10 = (num_vif_5to10 / total_vif) * 100
        pct_vif_gt10 = (num_vif_gt10 / total_vif) * 100

        return pct_vif_lt5, pct_vif_5to10, pct_vif_gt10
    else:
        return 0, 0, 0

vif_lasso = compute_vif(X, nonzero_features_lasso)
vif_ridge = compute_vif(X, nonzero_features_ridge)
vif_en = compute_vif(X, nonzero_features_en)

# Print VIF summary
print(f"\nVIF Summary for Lasso:\n% VIF < 5: {vif_lasso[0]:.2f}%\n% VIF 5-10: {vif_lasso[1]:.2f}%\n% VIF >= 10: {vif_lasso[2]:.2f}%")
print(f"\nVIF Summary for Ridge:\n% VIF < 5: {vif_ridge[0]:.2f}%\n% VIF 5-10: {vif_ridge[1]:.2f}%\n% VIF >= 10: {vif_ridge[2]:.2f}%")
print(f"\nVIF Summary for ElasticNet:\n% VIF < 5: {vif_en[0]:.2f}%\n% VIF 5-10: {vif_en[1]:.2f}%\n% VIF >= 10: {vif_en[2]:.2f}%")
