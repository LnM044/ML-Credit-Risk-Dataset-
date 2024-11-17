import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from scipy import stats

# Backend Preprocessing Function
@st.cache_data
def preprocess_dataset(file_name):
    df = pd.read_csv(file_name)

    # Move the "loan_status" column to the end
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

    # Encoding and Scaling
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

    # Encode target column
    df_encoded["loan_status"] = label_encoder.fit_transform(df_encoded["loan_status"])

    return df_encoded

# Load and preprocess the dataset
file_name = "credit_risk_dataset.csv"
df_encoded = preprocess_dataset(file_name)

if df_encoded is not None:
    # Streamlit Application: Predictions and Visualizations
    st.title("Credit Risk Dataset Analysis and Model Training")

    st.header("Dataset Overview")
    st.write("### First 5 Rows of Encoded and Scaled Data")
    st.dataframe(df_encoded.head())

    # Split the dataset into features and target variable
    X = df_encoded.drop("loan_status", axis=1)
    y_encoded = df_encoded["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    st.write(f"Training Set Size: {X_train.shape}")
    st.write(f"Test Set Size: {X_test.shape}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    # Evaluate models
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        )
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    # Evaluate and display results
    st.header("Model Evaluation")
    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(model, X_train, X_test, y_train, y_test)

    metrics_df = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]
    st.write("### Performance Metrics")
    st.dataframe(metrics_df)

    # ROC Curve
    st.header("ROC Curves")
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        if result["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {result['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

    # Confusion Matrix
    st.header("Confusion Matrices")
    for model_name, result in results.items():
        st.write(f"### {model_name}")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, result["y_pred"])
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)
