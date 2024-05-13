import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
def preprocess_data():
   df = pd.read_csv('creditcard_2023.csv')
   df_clean = df.copy()
   df_clean.drop('id', axis=1, inplace=True)
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
def decision_tree(X,y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   # Reload the trained classifier from the file
   with open("trained_decisionTree.pkl", "rb") as f:
      clf_loaded = pickle.load(f)

   # Make predictions using the loaded classifier (optional)
   y_pred = clf_loaded.predict(X_test)
   return y_pred

def gradient_boosting(X,y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100000, test_size=100000, random_state=42)
   with open("trained_gradientBoosting.pkl", "rb") as f:
      clf_loaded = pickle.load(f)

   # Make predictions using the loaded classifier (optional)
   y_pred = clf_loaded.predict(X_test)
   return y_pred

def random_forest(X,y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100000, test_size=100000, random_state=42)
   with open("trained_gradientBoosting.pkl", "rb") as f:
      clf_loaded = pickle.load(f)

   # Make predictions using the loaded classifier (optional)
   y_pred = clf_loaded.predict(X_test)
   return y_pred

X,y = preprocess_data()
decision_tree_pred = decision_tree(X, y)
gradient_boosting_pred = gradient_boosting(X, y)
# random_forest_pred = random_forest(X,y)

def run_models():
   print("Decision_tree_pred", decision_tree_pred)
   print("Gradient_Boosting_pred", gradient_boosting_pred)
   print(len(decision_tree_pred))
   #print(random_forest_pred)

st.title('Credit Card Fraud Detection')
st.write("Our app will help you detect any fraudulent transactions!")
st.divider()

container = st.container(border=True)
with container:
   st.write("Click one of these buttons to check for fraudulent transactions")
   col1, col2 = st.columns(2)
   with col1:
      if st.button("Check 'Card 1' for Fraudulent Transactions", key="Card1"):
         print("printed!")
   with col2:
      if st.button("Check 'Card 2' for Fraudulent Transactions", key="Card2"):
         print("printed! 2")

