import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
st.title("Credit Risk Dataset Analysis and Visualization")

# Backend processing (hidden from the user)
@st.cache_data
def load_and_preprocess_data(file_name):
    try:
        df = pd.read_csv(file_name)

        # Moving "loan_status" column to the end
        column_to_move = "loan_status"
        column_data = df.pop(column_to_move)
        df[column_to_move] = column_data

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df_cleaned = df.dropna().reset_index(drop=True)

        # Outlier detection and removal
        def find_outliers_zscore_mask(data, threshold=3):
            z_scores = np.abs(stats.zscore(data))
            return z_scores > threshold

        age_zscore = find_outliers_zscore_mask(df_cleaned['person_age'])
        df_age_outliers_removed = df_cleaned[~age_zscore]

        income_zscore = find_outliers_zscore_mask(df_age_outliers_removed['person_income'])
        df_income_outliers_removed = df_age_outliers_removed[~income_zscore]

        amount_zscore = find_outliers_zscore_mask(df_income_outliers_removed['loan_amnt'])
        df_removed = df_income_outliers_removed[~amount_zscore]

        # Apply Encoding and Scaling
        categorical_columns_label = ['cb_person_default_on_file', 'person_home_ownership', 'loan_grade']
        categorical_columns_onehot = ['loan_intent']
        numerical_columns = df_removed.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Apply Standard Scaling to numerical features
        scaler = StandardScaler()
        df_removed[numerical_columns] = scaler.fit_transform(df_removed[numerical_columns])

        # Apply one-hot encoding for nominal features
        df_encoded = pd.get_dummies(df_removed, columns=categorical_columns_onehot, drop_first=True)

        # Apply label encoding to ordinal or binary features
        label_encoder = LabelEncoder()
        for col in categorical_columns_label:
            df_encoded[col] = label_encoder.fit_transform(df_removed[col])

        return df, df_cleaned, df_removed, df_encoded, numerical_columns
    except FileNotFoundError:
        st.error(f"File '{file_name}' not found. Please ensure it is in the same directory.")
        return None, None, None, None, None

# Load and preprocess the dataset
file_name = "credit_risk_dataset.csv"
df, df_cleaned, df_removed, df_encoded, numerical_columns = load_and_preprocess_data(file_name)

if df is not None:
    # Frontend Visualizations and Interactions
    st.header("Dataset Overview")
    if st.checkbox("Show Dataset Preview"):
        st.write("### Dataset Preview")
        st.dataframe(df.head())

    if st.checkbox("Show Missing Values Summary"):
        st.write("### Missing Values Summary")
        st.write(df.isnull().sum())

    if st.checkbox("Show Cleaned Data Preview"):
        st.write("### Cleaned Data Preview")
        st.dataframe(df_cleaned.head())

    st.header("Loan Status Distribution")
    st.bar_chart(df_cleaned["loan_status"].value_counts())

    if st.checkbox("Show Dataset Statistics"):
        st.write("### Statistical Summary")
        st.write(df_cleaned.describe())

    if st.checkbox("Show Outlier Removed Data Preview"):
        st.write("### Outlier Removed Data Preview")
        st.dataframe(df_removed.head())

    st.header("Encoded and Scaled Data")
    if st.checkbox("Show Encoded and Scaled Data Preview"):
        st.dataframe(df_encoded.head())

    # Distribution of numerical features
    st.header("Distribution of Numerical Features")
    for column in numerical_columns:
        st.write(f"### Distribution of {column}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_encoded[column], kde=True, bins=30, color='blue', edgecolor='black', alpha=0.7, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # Correlation Matrix Heatmap
    st.header("Correlation Matrix Heatmap")
    correlation_matrix = df_encoded.corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask the upper triangle
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Half-Triangle Correlation Matrix of Encoded Data')
    st.pyplot(fig)
