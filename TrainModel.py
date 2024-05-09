from sklearn.ensemble import IsolationForest
from sklearn.tree import plot_tree
from sklearn.tree import _tree

import pandas as pd
import PreprocessData
# Global outlier detection by class


df_clean = PreprocessData.df_clean

isolation_forest = IsolationForest(contamination=0.01)
outlier_scores = isolation_forest.fit_predict(df_clean[feature_names])
df_clean['outlier'] = outlier_scores

outlier_counts = df_clean.groupby(['Class', 'outlier']).size().unstack()
tree = isolation_forest.estimators_[0]

