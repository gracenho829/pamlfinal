import streamlit as st
import pandas as pd
import numpy as np

st.title('Credit Card Fraud Detection')
st.write("Our app will help you detect any fraudulent transactions!")
st.divider()

container = st.container(border=True)
col1, col2, col3 = st.columns(3)
with container:
   st.write("Click one of these buttons to check for fraudulent transactions")
   with col1:
      st.button("Check 'Card 1' for Fraudulent Transactions", key="Card1", on_click=None)
   with col2:
      st.button("Check 'Card 2' for Fraudulent Transactions", key="Card2", on_click=None)
   with col3:
      st.button("Check 'Card 3' for Fraudulent Transactions", key="Card3", on_click=None)

