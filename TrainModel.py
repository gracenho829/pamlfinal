from sklearn.ensemble import IsolationForest
from sklearn.tree
import numpy as np
import pandas as pd
import PreprocessData
# Global outlier detection by class
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import _tree


df_clean = PreprocessData.df_clean
feature_names = PreprocessData.feature_names

isolation_forest = IsolationForest(contamination=0.01)
outlier_scores = isolation_forest.fit_predict(df_clean[feature_names])
df_clean['outlier'] = outlier_scores

outlier_counts = df_clean.groupby(['Class', 'outlier']).size().unstack()
tree = isolation_forest.estimators_[0]

def find_paths_to_outliers_df(tree, node_id=0, path=[]):
    if tree.tree_.feature[node_id] == _tree.TREE_UNDEFINED:
        if len(path) <= 8:
            return [dict(path)]
        else:
            return []

    feature = tree.tree_.feature[node_id]
    threshold = tree.tree_.threshold[node_id]
    left_paths = find_paths_to_outliers_df(tree, tree.tree_.children_left[node_id],
                                           path + [(feature_names[feature] + " <= ", threshold)])
    right_paths = find_paths_to_outliers_df(tree, tree.tree_.children_right[node_id],
                                            path + [(feature_names[feature] + " > ", threshold)])
    return left_paths + right_paths

outlier_paths_dicts = find_paths_to_outliers_df(tree)
df_outlier_paths = pd.DataFrame(outlier_paths_dicts)
df_processed = df_outlier_paths.notnull().astype('int')
condition_frequencies = df_processed.sum().sort_values(ascending=False)
outlier_scores = isolation_forest.decision_function(df_clean[feature_names])


X = df_clean[feature_names]
y = df_clean['Class']

model = ExtraTreesClassifier(n_estimators=100, random_state=42)

model.fit(X, y)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]