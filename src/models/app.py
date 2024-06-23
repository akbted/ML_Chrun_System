import streamlit as st
import pandas as pd
from Gboost_tuning import loaded_model  # Assuming loaded_model returns the trained model
from joblib import load

loaded_model = load("Gboost_my_churn_model.pkl")

# ## Input fields for feature values on the main screen
# st.header("Enter Customer Information")
# tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
# internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
# contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
# monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
# total_charges = st.number_input("Total Charges", min_value=0, max_value=10000, value=0)

# # Map input values to numeric (if label encoding is required)
# label_mapping = {
#     'DSL': 0,
#     'Fiber optic': 1,
#     'No': 2,
#     'Month-to-month': 0,
#     'One year': 1,
#     'Two year': 2,
# }
# internet_service = label_mapping[internet_service]
# contract = label_mapping[contract]

# # Make a prediction using the model (replace with your actual prediction logic)
# try:
#   prediction = loaded_model.predict([[tenure, internet_service, contract, monthly_charges, total_charges]])
#   # Display the prediction result on the main screen
#   st.header("Prediction Result")
#   if prediction[0] == 0:
#       st.success("This customer is likely to stay.")
#   else:
#       st.error("This customer is likely to churn.")
# except Exception as e:  # Add specific exception handling if needed
#   st.error(f"An error occurred: {e}")



# import streamlit as st

# # Set a page layout and background color
# st.set_page_config(layout="wide", page_title="Churn Prediction App")
# # st.beta_set_background_color("#f0f2f5")

# # Header with a centered title and a brief description
# st.title("Customer Churn Prediction")
# st.subheader("Enter customer information to predict churn likelihood.")

# # Create columns for better layout
# col1, col2, col3 = st.columns(3)

# # Tenure as a slider input field
# with col1:
#     tenure = st.slider("Tenure (Months)", min_value=0, max_value=100, value=1)

# with col2:
#     internet_service = st.selectbox("InternetService", ('DSL', 'Fiber optic', 'No'))
# with col3:
#     contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))

# # Input fields with adjusted ranges and a color for visual appeal
# col1, col2 = st.columns(2)
# with col1:
#     monthly_charges = st.number_input("MonthlyCharges (₹)", min_value=0, max_value=200, value=50, key="monthly_charges")
# with col2:
#     total_charges = st.number_input("TotalCharges (₹)", min_value=0, max_value=10000, value=0, key="total_charges")


# # Label encoding for categorical features (hidden from user)
# label_mapping = {
#     'DSL': 0,
#     'Fiber optic': 1,
#     'No': 2,
#     'Month-to-month': 0,
#     'One year': 1,
#     'Two year': 2,
# }
# internet_service = label_mapping[internet_service]
# contract = label_mapping[contract]

# # Make a prediction using the model (replace with your actual prediction logic)
# try:
#   prediction = loaded_model.predict([[tenure, internet_service, contract, monthly_charges, total_charges]])

#   # Display the prediction result with success/error color and centered text
#   st.header("Prediction Result")
#   if prediction[0] == 0:
#       st.success("This customer is likely to stay.")
#   else:
#       st.error("This customer is likely to churn.")
# except Exception as e:  # Add specific exception handling if needed
#   st.error(f"An error occurred: {e}")




##TEST 3


# import streamlit as st

# # Set a page layout and background color
# st.set_page_config(layout="wide", page_title="Churn Prediction App")
# # st.beta_set_background_color("#f0f2f5")

# # Header with a centered title and a brief description
# st.title("Customer Churn Prediction")
# st.subheader("Enter customer information to predict churn likelihood.")

# # Create columns for better layout
# col1, col2, col3 = st.columns(3)

# # Tenure as a slider input field
# with col1:
#     tenure = st.slider("Tenure (Months)", min_value=0, max_value=100, value=1)

# with col2:
#     InternetService = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No Service'))
#     MultipleLines  = st.selectbox("MultipleLines Status  :telephone_receiver:", ("Yes", "No", "No phone service"))
#     Dependents = st.radio("Do you have any dependents? :baby:", ("Yes", "No"))
#     PaperlessBilling = st.radio("Billing type - PaperlessBilling :inbox_tray:", ("Yes", "No"))
#     PaymentMethod = st.selectbox("Payment Method  :credit_card:", ("Electronic check", "Mailed check","Bank transfer (automatic)","Credit card (automatic)"))
# with col3:
#     Contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
#     TechSupport = st.selectbox("Tech Support Status  :headphones:", ("Available", "Not Available", "No internet service"))
#     SeniorCitizen = st.radio("Senior Citizen Status :", ("Yes", "No"))
#     Partner = st.radio("Relationship Status :family:", ("Married", "Single"))
   


# # Input fields with adjusted ranges and a color for visual appeal
# col1, col2 = st.columns(2)
# with col1:
#     MonthlyCharges = st.number_input("Monthly Charges (₹)", min_value=0, max_value=200, value=50, key="monthly_charges")
# with col2:
#     TotalCharges = st.number_input("Total Charges (₹)", min_value=0, max_value=10000, value=0, key="total_charges")

# # Button to trigger prediction (currently disabled)
# predict_button = st.button("Predict")

# # Label encoding for categorical features (hidden from user)
# label_mapping = {
#     'DSL': 0,
#     'Fiber optic': 1,
#     'No Service': 2,
#     'Month-to-month': 0,
#     'One year': 1,
#     'Two year': 2,
#     'Yes' : 1, 
#     'No' : 0, 
#     'No phone service' : 2,
#     'No internet service': 2,
#     'Electronic check' : 0, 
#     'Mailed check ' : 1, 
#     'Bank transfer (automatic)' : 2, 
#     'Credit card (automatic)' : 3,
#     'Married' : 1, 
#     'Single' : 0,
#     "Available" : 1, 
#     "Not Available" : 0, 
# }

# # Perform prediction only when the button is clicked
# if predict_button:
#   InternetService = label_mapping[InternetService]
#   MultipleLines = label_mapping[MultipleLines]
#   Dependents = label_mapping[Dependents]
#   Partner = label_mapping[Partner]
#   PaperlessBilling = label_mapping[PaperlessBilling]
#   PaymentMethod = label_mapping[PaymentMethod]
#   TechSupport = label_mapping[TechSupport]
#   SeniorCitizen = label_mapping[SeniorCitizen]

#   Contract = label_mapping[Contract]


#   try:
#     prediction = loaded_model.predict([[tenure, InternetService, Contract, MonthlyCharges, TotalCharges, TotalCharges, Dependents, PaperlessBilling, PaymentMethod, SeniorCitizen, Partner, TechSupport, MultipleLines]])

#     # Display the prediction result with success/error color and centered text
#     st.header("Prediction Result")
#     if prediction[0] == 0:
#         st.success("This customer is likely to stay.")
#     else:
#         st.error("This customer is likely to churn.")
#   except Exception as e:  # Add specific exception handling if needed
#     st.error(f"An error occurred: {e}")




##Test 4


import streamlit as st

# Set a page layout and background color
st.set_page_config(layout="wide", page_title="Churn Prediction App")
# st.beta_set_background_color("#f0f2f5")

# Header with a centered title and a brief description
st.title("Customer Churn Prediction")
st.subheader("Enter customer information to predict churn likelihood.")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

# Tenure as a slider input field
with col1:
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=100, value=1)

with col2:
    InternetService = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No Service'))
    MultipleLines = st.selectbox("MultipleLines Status  :telephone_receiver:", ("Yes", "No", "No phone service"))
    Dependents = st.radio("Do you have any dependents? :baby:", ("Yes", "No"))
    PaperlessBilling = st.radio("Billing type - PaperlessBilling :inbox_tray:", ("Yes", "No"))
    PaymentMethod = st.selectbox("Payment Method  :credit_card:", ("Electronic check", "Mailed check","Bank transfer (automatic)","Credit card (automatic)"))
with col3:
    Contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
    TechSupport = st.selectbox("Tech Support Status  :headphones:", ("Available", "Not Available", "No internet service"))
    SeniorCitizen = st.radio("Senior Citizen Status :", ("Yes", "No"))
    Partner = st.radio("Relationship Status :family:", ("Married", "Single"))
   

# Input fields with adjusted ranges and a color for visual appeal
col1, col2 = st.columns(2)
with col1:
    MonthlyCharges = st.number_input("Monthly Charges (₹)", min_value=0, max_value=200, value=50, key="monthly_charges")
with col2:
    TotalCharges = st.number_input("Total Charges (₹)", min_value=0, max_value=10000, value=0, key="total_charges")

# Button to trigger prediction (currently disabled)
predict_button = st.button("Predict")

# Label encoding for categorical features (hidden from user)
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No Service': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
    'Yes' : 1, 
    'No' : 0, 
    'No phone service' : 2,
    'No internet service': 2,
    'Electronic check' : 0, 
    'Mailed check' : 1, 
    'Bank transfer (automatic)' : 2, 
    'Credit card (automatic)' : 3,
    'Married' : 1, 
    'Single' : 0,
    "Available" : 1, 
    "Not Available" : 0, 
}

# Perform prediction only when the button is clicked
if predict_button:
    InternetService = label_mapping[InternetService]
    MultipleLines = label_mapping[MultipleLines]
    Dependents = label_mapping[Dependents]
    Partner = label_mapping[Partner]
    PaperlessBilling = label_mapping[PaperlessBilling]
    PaymentMethod = label_mapping[PaymentMethod]
    TechSupport = label_mapping[TechSupport]
    SeniorCitizen = label_mapping[SeniorCitizen]
    Contract = label_mapping[Contract]

    try:
        prediction = loaded_model.predict([[tenure, InternetService, Contract, MonthlyCharges, TotalCharges, Dependents, PaperlessBilling, PaymentMethod, SeniorCitizen, Partner, TechSupport, MultipleLines]])

        # Display the prediction result with success/error color and centered text
        st.header("Prediction Result")
        if prediction[0] == 0:
            st.success("This customer is likely to stay.")
        else:
            st.error("This customer is likely to churn.")
    except Exception as e:  # Add specific exception handling if needed
        st.error(f"An error occurred: {e}")