from pathlib import Path
from datetime import datetime, timezone
import os
import json
import requests
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "loan_model.joblib"
SHEETS_CREDENTIALS = ROOT / "service_account.json"

st.set_page_config(page_title="AI Loan Eligibility Checker", page_icon="🏦", layout="wide")

# Visual styling
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #172554 0, #0b1020 45%, #070b14 100%); }
.block-container { max-width: 1250px; padding-top: 2rem; }
.hero { padding: 1.5rem 1.8rem; border: 1px solid #ffffff22; border-radius: 22px;
        background: linear-gradient(120deg, #172554cc, #312e81aa); margin-bottom: 1rem; }
.glass { border: 1px solid #ffffff22; border-radius: 16px; padding: 1rem; background: #ffffff0a; }
.small-muted { color: #b7c4dc; font-size: .92rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🏦 AI Loan Eligibility Checker</h1>
<p class="small-muted">Loan prediction · EMI planning · credit profile · affordability insights</p>
</div>
""", unsafe_allow_html=True)
st.warning("Educational prototype only. The bundled dataset is synthetic; predictions are not real bank decisions or financial advice.")

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
if bundle is None:
    st.error("Model file is missing. Run `python train_model.py` in the project terminal, then refresh this page.")
    st.stop()

model = bundle["model"]
feature_names = bundle["features"]

def calculate_emi(principal, annual_rate, months):
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 1200
    if r == 0:
        return principal / months
    factor = (1+r) ** months
    return principal * r * factor / (factor - 1)

def claude_guidance(profile):
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
        except Exception:
            api_key = ""
    if not api_key:
        return None, "Claude is not configured yet. Add ANTHROPIC_API_KEY as an environment variable or Streamlit secret."
    prompt = (
        "Give concise, cautious educational financial planning guidance based on this profile. "
        "Do not guarantee loan approval, invent credit bureau data, or provide regulated personalized advice. "
        "Mention affordability and uncertainties. Profile JSON: " + json.dumps(profile)
    )
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-5-haiku-latest",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=25
        )
        response.raise_for_status()
        data = response.json()
        answer = "\n".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        return answer or "Claude returned no text.", None
    except Exception as exc:
        return None, f"Claude API request failed: {exc}. Check your key, account access, and network."

def save_to_google_sheets(record):
    # Optional connector; service account JSON and spreadsheet ID must be configured.
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
    try:
        if not spreadsheet_id:
            spreadsheet_id = st.secrets.get("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
    except Exception:
        pass
    if not spreadsheet_id:
        return False, "Google Sheets is not configured. Set GOOGLE_SHEETS_SPREADSHEET_ID and add service-account credentials."
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_dict = None
        try:
            creds_dict = st.secrets.get("gcp_service_account")
        except Exception:
            creds_dict = None
        if creds_dict:
            creds = Credentials.from_service_account_info(
                dict(creds_dict),
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
        elif SHEETS_CREDENTIALS.exists():
            creds = Credentials.from_service_account_file(
                str(SHEETS_CREDENTIALS),
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
        else:
            return False, "Service account credentials are missing. Add service_account.json locally or configure gcp_service_account in Streamlit secrets."
        client = gspread.authorize(creds)
        sheet = client.open_by_key(spreadsheet_id).sheet1
        if not sheet.get_all_values():
            sheet.append_row(list(record.keys()))
        sheet.append_row([str(v) for v in record.values()])
        return True, "Record saved to Google Sheets."
    except Exception as exc:
        return False, f"Google Sheets save failed: {exc}"

with st.sidebar:
    st.header("Project modules")
    st.markdown("✅ ML prediction\n\n✅ EMI calculator\n\n✅ Credit score insights\n\n✅ Affordability analysis\n\n🔐 Claude API (requires key)\n\n🔐 Google Sheets (requires setup)")
    st.caption("Never commit API keys or service-account credentials to GitHub.")

tab1, tab2, tab3, tab4 = st.tabs(["Eligibility & profile", "EMI calculator", "AI guidance", "Data storage"])

with tab1:
    st.subheader("Applicant details")
    with st.form("profile_form"):
        a,b,c = st.columns(3)
        with a:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=29)
            income = st.number_input("Applicant monthly income (₹)", min_value=0, max_value=10000000, value=75000, step=5000)
            co_income = st.number_input("Co-applicant monthly income (₹)", min_value=0, max_value=10000000, value=25000, step=5000)
            loan_amount = st.number_input("Loan amount requested (₹)", min_value=10000, max_value=100000000, value=1500000, step=50000)
        with b:
            credit_score = st.number_input("Credit score (user-entered)", min_value=300, max_value=900, value=740, step=5)
            loan_term = st.selectbox("Loan term (months)", [12,24,36,60,84,120,180,240,300,360], index=9)
            interest_rate = st.number_input("Annual interest rate (%)", min_value=0.0, max_value=40.0, value=9.0, step=0.25)
            employment = st.selectbox("Employment status", ["Salaried", "Self-Employed", "Unemployed"])
        with c:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
            dependents = st.selectbox("Number of dependents", ["0", "1", "2", "3+"])
            consent = st.checkbox("I consent to submit this profile to external AI or Google Sheets only when I click those actions.")
        submitted = st.form_submit_button("Analyze eligibility", type="primary", use_container_width=True)

    if submitted:
        st.session_state["profile"] = {
            "Age": int(age),
            "ApplicantIncome": int(income),
            "CoapplicantIncome": int(co_income),
            "CreditScore": int(credit_score),
            "LoanAmount": int(loan_amount),
            "Loan_Amount_Term": int(loan_term),
            "InterestRate": float(interest_rate),
            "Employment_Status": employment,
            "Education": education,
            "Property_Area": property_area,
            "Dependents": dependents
        }
        st.session_state["consent"] = bool(consent)

    profile = st.session_state.get("profile")
    if profile:
        input_df = pd.DataFrame([{k: profile[k] for k in feature_names}])
        pred = int(model.predict(input_df)[0])
        probs = model.predict_proba(input_df)[0]
        approval_score = float(probs[1])
        emi = calculate_emi(profile["LoanAmount"], profile["InterestRate"], profile["Loan_Amount_Term"])
        total_income = profile["ApplicantIncome"] + profile["CoapplicantIncome"]
        emi_ratio = emi / total_income if total_income else float("inf")
        surplus = total_income - emi
        st.divider()
        left, mid, right = st.columns(3)
        left.metric("Model classification", "Likely eligible" if pred == 1 else "May not be eligible")
        mid.metric("Model score for eligible class", f"{approval_score:.1%}")
        right.metric("Estimated monthly EMI", f"₹{emi:,.0f}")
        x,y,z = st.columns(3)
        x.metric("Total repayment", f"₹{emi*profile['Loan_Amount_Term']:,.0f}")
        y.metric("Estimated total interest", f"₹{max(0, emi*profile['Loan_Amount_Term']-profile['LoanAmount']):,.0f}")
        z.metric("EMI / combined income", "N/A" if not total_income else f"{emi_ratio:.1%}")
        if total_income == 0:
            st.error("Combined income is zero; affordability ratio cannot be calculated.")
        elif emi_ratio > 0.4:
            st.warning("Affordability flag: estimated EMI exceeds 40% of combined monthly income. Consider reviewing the amount or term.")
        else:
            st.success("Affordability flag: estimated EMI is at or below 40% of combined monthly income. Other expenses and debts still matter.")
        if profile["Age"] < 21 or profile["Age"] > 60:
            st.info("Age can affect lender-specific eligibility rules. Check the actual lender's criteria.")
        if profile["CreditScore"] >= 750:
            st.info("General credit-score interpretation: strong range in many scoring contexts; lender cutoffs vary.")
        elif profile["CreditScore"] >= 650:
            st.info("General credit-score interpretation: middle range; review repayment history and lender criteria.")
        else:
            st.warning("General credit-score interpretation: lower range; review credit reports and consider improving repayment history.")
        if st.button("Prepare record for optional storage", disabled=not st.session_state.get("consent", False)):
            st.session_state["record"] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **profile,
                "model_prediction": "Likely eligible" if pred == 1 else "May not be eligible",
                "model_score": round(approval_score, 4),
                "estimated_emi": round(emi, 2)
            }
            st.success("Record prepared. Go to the Data storage tab and click Save to Google Sheets.")

with tab2:
    st.subheader("EMI calculator")
    c1,c2,c3 = st.columns(3)
    with c1:
        p = st.number_input("Principal (₹)", min_value=1000, max_value=100000000, value=1000000, step=50000, key="emi_p")
    with c2:
        annual = st.number_input("Annual interest rate (%)", min_value=0.0, max_value=40.0, value=9.0, step=0.25, key="emi_rate")
    with c3:
        months = st.number_input("Tenure (months)", min_value=1, max_value=600, value=60, step=1, key="emi_months")
    emi_value = calculate_emi(p, annual, months)
    st.metric("Monthly EMI", f"₹{emi_value:,.2f}")
    st.metric("Total repayment", f"₹{emi_value*months:,.2f}")
    st.metric("Total interest", f"₹{max(0, emi_value*months-p):,.2f}")

with tab3:
    st.subheader("Claude AI financial guidance")
    st.caption("Requires a valid Anthropic API key and network access. Profile is sent to Anthropic only when you click the button.")
    profile = st.session_state.get("profile")
    if not profile:
        st.info("Submit applicant details in the Eligibility & profile tab first.")
    elif not st.session_state.get("consent", False):
        st.warning("Tick the consent checkbox before sending any profile to an external AI service.")
    elif st.button("Ask Claude for guidance"):
        with st.spinner("Contacting Claude..."):
            answer, error = claude_guidance(profile)
        if error:
            st.error(error)
        else:
            st.markdown(answer)

with tab4:
    st.subheader("Google Sheets record storage")
    st.caption("Requires Google Cloud service-account setup and a spreadsheet shared with that service-account email.")
    record = st.session_state.get("record")
    if not record:
        st.info("Analyze a profile, tick consent, and click 'Prepare record for optional storage' to create a record.")
    elif not st.session_state.get("consent", False):
        st.warning("Consent is required before saving records.")
    elif st.button("Save to Google Sheets"):
        ok, message = save_to_google_sheets(record)
        (st.success if ok else st.error)(message)

with st.expander("Methodology and limitations"):
    st.write("The included model is a Logistic Regression classifier trained on synthetic demonstration data. The displayed score is a model output, not a calibrated real-world probability of approval.")
    st.write("Age and credit score are included as model inputs in this demo. Actual lender policies vary; use verified credit bureau information and a representative, lawfully obtained dataset for meaningful validation.")
    st.write("External API integrations are optional and require your own credentials. Never expose keys in source code or commit credentials to version control.")
