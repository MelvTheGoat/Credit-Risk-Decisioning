import streamlit as st
import requests
import uuid

# Points to FastAPI running internally on port 8000 inside the same container
API_URL = "http://127.0.0.1:8000/decision"

st.set_page_config(page_title="Credit Risk Portal", page_icon="🏦", layout="centered")

st.title("🏦 Credit Risk Decisioning Portal")
st.markdown("Evaluate applicant risk profiles against the deployed LightGBM model.")

with st.form("application_form"):
    st.subheader("Applicant Features")
    
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income (₦)", min_value=0, value=12000000, step=500000)
        loan_amount = st.number_input("Loan Amount (₦)", min_value=0, value=2500000, step=100000)
        employment_years = st.number_input("Employment Years", min_value=0, value=5)
        age = st.number_input("Applicant Age", min_value=18, value=32)
        housing = st.selectbox("Housing Status", ["rent", "own", "mortgage", "other"])
        
    with col2:
        dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, value=15.0)
        utilisation = st.number_input("Credit Utilisation (%)", min_value=0.0, value=12.0)
        delinq = st.number_input("Delinquencies (Last 24m)", min_value=0, value=0)
        history = st.number_input("Credit History (Months)", min_value=0, value=48)
        inquiries = st.number_input("Recent Inquiries (6m)", min_value=0, value=1)
        product = st.selectbox("Product Type", ["personal_loan", "auto", "card"])

    submit = st.form_submit_button("Evaluate Application", type="primary")

if submit:
    payload = {
        "application_id": f"APP-{uuid.uuid4().hex[:8].upper()}",
        "features": {
            "income_recorded": income,
            "debt_to_income": dti,
            "utilisation": utilisation,
            "n_delinq_24m": delinq,
            "credit_history_months": history,
            "n_inquiries_6m": inquiries,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "age": age,
            "housing_status": housing,
            "product_type": product
        },
        "exposure_at_default": loan_amount
    }
    
    with st.spinner("Scoring applicant..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                st.success("Decision Processed Successfully!")
                st.json(result)
            else:
                st.error(f"API Error {response.status_code}")
                st.json(response.json())
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
