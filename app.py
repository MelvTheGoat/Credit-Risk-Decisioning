import streamlit as st
import requests
import uuid

# Points to FastAPI running internally on port 8000
API_URL = "http://127.0.0.1:8000/decision"

st.set_page_config(page_title="Credit Risk Portal", page_icon="🏦", layout="centered")

st.title("🏦 Credit Risk Decisioning Portal")
st.markdown("Evaluate applicant risk profiles against the deployed LightGBM model.")

with st.form("application_form"):
    st.subheader("Applicant Features")
    
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income (₦)", min_value=0, max_value=10000000, value=5000000, step=250000)
        loan_amount = st.number_input("Loan Amount (₦)", min_value=0, value=2500000, step=100000)
        employment_years = st.number_input("Employment Years", min_value=0, value=5)
        age = st.number_input("Applicant Age", min_value=18, value=30)
        housing = st.selectbox("Housing Status", ["rent", "own", "mortgage", "other"])
        
    with col2:
        dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
        utilisation = st.number_input("Credit Utilisation", min_value=0.0, max_value=5.0, value=1.2, step=0.1)
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
                
                st.divider()
                
                # 1. MAIN DECISION BANNER
                decision = result.get("decision", "").upper()
                if decision == "APPROVE":
                    st.success(f"### ✅ DECISION: {decision}")
                else:
                    st.error(f"### ❌ DECISION: {decision}")
                
                # 2. KEY METRICS ROW
                col_a, col_b, col_c = st.columns(3)
                
                score = result.get('score', 0)
                band = result.get('score_band', 'N/A')
                col_a.metric(label="Credit Score", value=f"{score:.0f}", delta=f"Band {band}", delta_color="off")
                
                pd = result.get('probability_of_default', 0) * 100
                col_b.metric(label="Probability of Default", value=f"{pd:.1f}%")
                
                expected_loss = result.get('expected_loss', 0)
                col_c.metric(label="Expected Loss", value=f"₦ {expected_loss:,.2f}")
                
                # 3. REASON CODES (HUMAN READABLE)
                reasons = result.get("reason_codes", [])
                if reasons:
                    st.markdown("#### Key Factors Influencing Decision:")
                    for r in reasons:
                        feature_name = r.get('feature', '').replace('_', ' ').title()
                        statement = r.get('statement', '')
                        
                        # Flag protected classes (like age) visually
                        if r.get('protected_basis'):
                            st.warning(f"**{feature_name}**: {statement} *(Protected Basis)*")
                        else:
                            st.info(f"**{feature_name}**: {statement}")
                
                # 4. METADATA FOOTER
                st.divider()
                st.caption(f"**App ID:** {result.get('application_id')} | **Time:** {result.get('decided_at')} | **Model:** {result.get('model_version')}")

            else:
                st.error(f"API Validation Error {response.status_code}")
                st.json(response.json()) # Keep JSON here only for debugging bad inputs
                
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
