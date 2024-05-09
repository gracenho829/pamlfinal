import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import (
    shapiro,
    anderson,
    probplot,
    skew,
    chi2_contingency,
    mannwhitneyu,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

def find_outlier_indices(data_series, method='z-score', z_score_threshold=3,
    outlier_multiplier=1.5):
    if method == 'z-score':
        # Calculate the Z-scores for the data series
        z_scores = (data_series - data_series.mean()) / data_series.std()

        # Find the indices where the absolute Z-score exceeds the threshold
        outlier_indices = z_scores[abs(z_scores) > z_score_threshold].index.tolist()

        if len(outlier_indices) > 0:
            return {
                'index': outlier_indices,
                'z-score': z_scores[outlier_indices].round(2).tolist()
            }
        else:
            return None
    elif method == 'IQR':
        # Calculate quartiles
        Q1 = data_series.quantile(0.25)
        Q3 = data_series.quantile(0.75)

        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1

        # Calculate the lower and upper bounds for identifying outliers
        lower_bound = Q1 - outlier_multiplier * IQR
        upper_bound = Q3 + outlier_multiplier * IQR

        # Find the indices of outliers
        outlier_indices = data_series[
            (data_series < lower_bound) | (data_series > upper_bound)
        ].index.tolist()

        if len(outlier_indices) > 0:
            return {
                'index': outlier_indices,
                'IQR_diff': data_series[outlier_indices].apply(
                    lambda x: x - lower_bound if x < lower_bound else x - upper_bound
                ).tolist(),
            }
        else:
            return None


df = pd.read_csv('creditcard_2023.csv')
# created a copy of the raw df so that we can clean it
df_clean = df.copy()
# id is irrelevant to us, only the features really are
df_clean.drop('id', axis=1, inplace=True)
# converting the feature Class to a string since this is going to be our target (fraud or not fraud)
df_clean['Class'] = df_clean['Class'].astype(str)

target_name = ['Class']

feature_names = [
    name
    for name in df_clean.columns
    if name not in target_name
]

# dropping duplicates so that each row is unique
df_clean.drop_duplicates(subset=feature_names, inplace=True)

df_clean['Class'].value_counts()
# Create a dictionary to store outlier indices for each feature using the IQR method
outlier_mapping = {
    col: find_outlier_indices(df_clean[col], method='IQR')
    for col in feature_names
}

# Extract outlier indices from the dictionary and flatten the list
outlier_indices = sum(
    [item['index'] for _, item in outlier_mapping.items() if item is not None],
    []
)

# Print the number of outliers detected for each feature
print(f'Number of outliers per factor:')
pd.DataFrame(
    {
        col: [len(item['index'])] if item is not None else 0
        for col, item in outlier_mapping.items()
    }
).transpose() \
    .reset_index() \
    .rename(columns={'index': 'Feature', 0: 'Number of outlier detected'}) \
    .sort_values(['Number of outlier detected'], ascending=False)

def keys_with_value(dict, x):
    '''
    Get indices for outlier instances detected for each transaction.
    '''
    return [key for key, value in dict.items() if value == x]

def crosstab(df, indices, target):
    '''
    Generate a cross-tabulation of the target variable against whether
    each row index is in the specified list of indices.
    '''
    return pd.crosstab(df[target].values.flatten(), df.index.isin(list(set(indices))))

def frequency_count(df, indices, target):
    '''
    Compute the frequency count and chi-squared test p-value for a contingency table.
    '''
    try:
        crosstab = pd.crosstab(df[target].values.flatten(), df.index.isin(list(set(indices))))
        _, pvalue, *_ = chi2_contingency(crosstab)
        freq = crosstab.values[:, 1].tolist()
        return freq + [pvalue]
    except IndexError:
        return []

# Count occurrences of outliers
outlier_counter = Counter(outlier_indices)

# Create a DataFrame to store frequency counts and p-values
data=pd.DataFrame(
    [
        [i] + frequency_count(
            df=df_clean,
            indices=keys_with_value(outlier_counter, i),
            target=target_name,
        )
        for i in range(1, len(feature_names))
    ]
).rename(columns={0: 'Count', 1: 'Non-Fraud', 2: 'Fraud', 3: 'p-value'})

