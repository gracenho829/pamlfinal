import streamlit as st
import pandas as pd
import numpy as np

st.title('Credit Card Fraud Detection')
st.write("Our app will help you detect any fraudulent transactions!")
st.divider()

with st.form("fraudulent_transaction_form"):
   st.write("Put in information of your fraudulent transaction here. ")
   number = st.number_input("Insert a number", value=None, step=1, placeholder="Field 1")
   # Every form must have a submit button.
   submitted = st.form_submit_button("Check for Fraudulent Transaction")
   if submitted:
       st.write("Field 1", number)
