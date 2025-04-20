import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import time

# Load the model
with open('dt_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Set Streamlit page config
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="centered")

# App function
def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏦 Loan Approval Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("#### Fill out the details below to check your loan approval status:")

    # Split the inputs into columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=500)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
        loan_amount_term = st.number_input("Loan Amount Term", min_value=1, value=360)
        credit_history = st.selectbox("Credit History", [1, 0])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Collect the input into a DataFrame
    new_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    # Convert categorical features to numeric
    label_encoders = {}
    for column in new_data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        new_data[column] = le.fit_transform(new_data[column])
        label_encoders[column] = le

    st.markdown("---")

    # Predict button
    if st.button("🔍 Predict Loan Approval"):
        with st.spinner("Analyzing..."):
            # # Fake progress bar for animation effect
            # for i in range(100):
            #     time.sleep(0.01)
            #     st.progress(i + 1)

            # Make prediction
            prediction = loaded_model.predict(new_data)
            result = "✅ Approved" if prediction[0] == 'Y' else "❌ Rejected"

            # Show result
            st.success(f"**Loan Prediction Result:** {result}")
            if result == "✅ Approved":
                st.balloons()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4334/4334849.png", width=100)
    st.markdown("### Navigation")
    st.info("Use this app to check your eligibility for a loan based on your financial details.")

# Run the app
if __name__ == "__main__":
    main()
