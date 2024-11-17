import streamlit as st
from config import APP_TITLE, APP_ICON

####### Page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

# Main content
st.title("Credit Risk Dataset")

st.markdown("""
This is the Streamlit app which will show the model for predicting the will the bank give a loan or not based on different features.
Use the navigation on the left to:
1. Explore the data and visualize relationships.
2. Predict concrete strength interactively.

            

This dataset includes 32581 instances with 11 features and 1 target variable which can be used for binary classification problem. 
This dataset is new and only in 10th of November in 2024 was uploaded to the website kaggle.com
The official website of the dataset is: https://www.kaggle.com/datasets/rizqi01/ps4e9-original-data-loan-approval-prediction
            
            
Total Records: 32,581 entries, representing a diverse sample of credit-seeking individuals.
Missing Values: Certain columns, specifically person_emp_length and loan_int_rate, contain missing values, which may need to be addressed in preprocessing.

**Attribute Information:**

- **person_age:** measured in kg in a m3 mixture
- **person_income:** measured in kg in a m3 mixture
- **person_home_ownership:** measured in kg in a m3 mixture
- **person_emp_length:** measured in kg in a m3 mixture
- **loan_intent:** measured in kg in a m3 mixture
- **loan_grade:** measured in kg in a m3 mixture
- **loan_amnt:** measured in kg in a m3 mixture
- **loan_int_rate:** day (1~365)
- **loan_percent_income** measured in MPa
- **cb_person_default_on_file:** A categorical variable indicating if the individual has previously defaulted (Y for Yes, N for No). This historical record can be an essential predictor of future credit risk.
- **cb_person_cred_hist_length:** The length of the individual's credit history in years, with longer histories potentially indicating more experience with managing credit.
- **loan_status:** target variable

""")