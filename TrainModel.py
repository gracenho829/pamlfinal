from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import PreprocessData
# Global outlier detection by class
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import _tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

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

X = df_clean[feature_names]
y = df_clean['Class']

best_features = SelectKBest(score_func = f_classif,k = 'all')
fit = best_features.fit(X, y)

featureScores = pd.DataFrame(data = fit.scores_,index = list(X.columns),columns = ['ANOVA Score'])
featureScores = featureScores.sort_values(ascending = False,by = 'ANOVA Score')

from collections import defaultdict

list1 = ['V4', 'V14', 'V11', 'V12', 'V17', 'V3', 'V16', 'V10', 'V1', 'V18', 'V9', 'V2', 'V19', 'V6', 'V5', 'V7', 'V8', 'V21', 'V24', 'V13', 'V15', 'V26', 'V22', 'V25', 'V27', 'V20', 'V28', 'V23', 'Amount']
list2 = ['V14', 'V12', 'V4', 'V11', 'V3', 'V10', 'V9', 'V16', 'V1', 'V2', 'V7', 'V17', 'V6', 'V18', 'V5', 'V19', 'V27', 'V20', 'V8', 'V24', 'V21', 'V28', 'V13', 'V26', 'V25', 'V15', 'V22', 'V23', 'Amount']
list3 = ['V24', 'V4', 'V13', 'V10', 'V15', 'V23', 'V28', 'V19', 'V13', 'V11', 'V17', 'V7', 'V5', 'V9', 'V27', 'V16', 'V9', 'Amount', 'V3', 'V6', 'V25', 'V8', 'V12']

def calculate_scores(features):
    scores = defaultdict(int)
    for index, feature in enumerate(reversed(features), 1):
        scores[feature] += index
    return scores

scores1 = calculate_scores(list1)
scores2 = calculate_scores(list2)
scores3 = calculate_scores(list3)

total_scores = defaultdict(int)
for feature in set(list1 + list2 + list3):
    total_scores[feature] = scores1[feature] + scores2[feature] + scores3[feature]

sorted_features = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)

top_features = sorted_features[:10]
top_features = sorted_features[:15]