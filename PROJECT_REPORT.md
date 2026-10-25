# AI Loan Eligibility Checker — Project Report

## Abstract
This project demonstrates a web-based loan profile analysis tool that combines a supervised machine-learning classification pipeline with EMI calculation, affordability indicators, basic credit-score interpretation, and optional integrations with Claude AI and Google Sheets.

## Problem Statement
Applicants often need to understand loan costs and the factors that lenders may review. The project demonstrates how a web application can collect profile inputs, estimate loan repayments, and produce an educational classification. It does not replace a lender's underwriting decision.

## Objectives
1. Build an interactive loan eligibility prediction interface.
2. Use age, credit score, income, employment, loan amount, term and related fields as model inputs.
3. Calculate EMI, total repayment and total interest using a standard financial formula.
4. Display an affordability indicator and general credit-score insights.
5. Provide optional Claude API guidance and Google Sheets record storage.
6. Include consent checks, input validation and secret-handling guidance.

## Technology Stack
Python, Streamlit, Pandas, Scikit-learn, Logistic Regression, Joblib, Requests, Anthropic Messages API, Google Sheets API, gspread.

## Methodology
Data is loaded from CSV, numerical values are imputed and scaled, categorical values are imputed and one-hot encoded, and Logistic Regression is trained using a held-out test split. The complete preprocessing and model pipeline is saved with Joblib. EMI is calculated using the standard amortizing-loan formula. Affordability is summarized as estimated EMI divided by combined monthly income.

## System Modules
- Applicant input and validation
- Machine-learning prediction
- EMI calculator
- Credit score interpretation
- Affordability and financial risk indicators
- Optional external AI guidance
- Optional Google Sheets storage
- Documentation and testing

## External Integrations
Claude and Google Sheets require user-provided credentials and service configuration. The code is included, but these services cannot work until valid API credentials, account access, spreadsheet configuration and network connectivity are available. External calls are initiated only by user actions after consent.

## Dataset and Evaluation
The bundled CSV is synthetic demonstration data. Model evaluation metrics are printed by `python train_model.py`. Metrics from synthetic data demonstrate the pipeline only and should not be interpreted as real-world lending performance.

## Privacy and Ethics
No real applicant data should be used without a lawful basis and informed consent. API keys and service-account files must not be committed to public repositories. A model prediction is not a guaranteed approval decision; real lending systems require representative data, calibration, fairness assessment, explainability and compliance review.

## Conclusion
The project provides an end-to-end educational prototype covering data preprocessing, classification, repayment calculations, risk indicators, and optional external integrations. Future work includes validation on an appropriate public dataset, fairness and calibration tests, authentication, PDF reports and deployment hardening.
