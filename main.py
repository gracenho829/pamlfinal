import pickle
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def preprocess_data():
   df = pd.read_csv('creditcard.csv')
   df_clean = df.copy()
   df_clean['Class'] = df_clean['Class'].astype(str)
   target_name = ['Class']

   feature_names = [
      name
      for name in df_clean.columns
      if name not in target_name
   ]
   df_clean.drop_duplicates(subset=feature_names, inplace=True)
   X_preprocessed = df_clean.drop(columns=["Class"])
   y_preprocessed = df_clean["Class"]
   return X_preprocessed,y_preprocessed

X,y = preprocess_data()

st.title('Credit Card Fraud Detection')
st.write("Our app will help you detect any fraudulent transactions!")
st.divider()

container = st.container(border=True)
col1 = st.columns(1)
with container:
   st.write("Click one of these buttons to check for fraudulent transactions")
   with col1:
      st.button("Check 'Card 1' for Fraudulent Transactions", key="Card1", on_click=None)



def decision_tree(X,y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   # Reload the trained classifier from the file
   with open("trained_decisionTree.pkl", "rb") as f:
      clf_loaded = pickle.load(f)

   # Make predictions using the loaded classifier (optional)
   y_pred = clf_loaded.predict(X_test)

def gradient_boosting(X,y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100000, test_size=100000, random_state=42)
   with open("trained_gradientBoosting.pkl.pkl", "rb") as f:
      clf_loaded = pickle.load(f)

   # Make predictions using the loaded classifier (optional)
   y_pred = clf_loaded.predict(X_test)



