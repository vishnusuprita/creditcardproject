import streamlit as st
import pandas as pd

# Set Streamlit page title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Title and Intro
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("""
Welcome! This app allows you to explore a credit card transactions dataset and use machine learning to detect fraudulent transactions.  
Below is a preview of the dataset along with a brief description of its columns.
""")

@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

df = load_data()

st.subheader("🔍 Sample of Dataset")
st.dataframe(df.head(10))

with st.expander("📘 About the Dataset & Columns"):
    st.markdown("""
    This dataset contains credit card transactions made by European cardholders in September 2013.  
    The dataset presents transactions that occurred over two days, with **284,807** transactions in total.  
    Among them, **492** are frauds — the dataset is highly imbalanced.

    - **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
    - **V1 to V28**: The result of PCA transformation to protect sensitive information.
    - **Amount**: Transaction amount.
    - **Class**: Target variable — `0` for legitimate, `1` for fraudulent.
    """)

st.subheader("📊 Dataset Summary")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Transactions", df.shape[0])
    st.metric("Total Features", df.shape[1])

with col2:
    st.bar_chart(df['Class'].value_counts(), use_container_width=True)
    st.caption("0 = Legitimate | 1 = Fraudulent")

st.markdown("👉 Use the sidebar to switch to prediction or model comparison.")

