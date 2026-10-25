# AI Loan Eligibility Checker — Full Feature Demo

## Run locally (Windows / VS Code)
Open this folder in VS Code. In Terminal → New Terminal, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. If you already have an environment, activate it and install the requirements.

## Included features
- Logistic Regression loan-class prediction with age, user-entered credit score, income, employment and loan details.
- EMI calculator, total repayment and interest calculations.
- Basic affordability indicator (EMI / combined monthly income).
- General credit-score interpretation.
- Claude API integration code (requires `ANTHROPIC_API_KEY` and an active Anthropic API account).
- Google Sheets integration code (requires spreadsheet ID, Google service-account credentials, and spreadsheet sharing).
- Consent gates before external AI calls or saving a record.
- Responsive Streamlit interface and form validation.
- Project report and documentation.

## Optional Claude setup
PowerShell (current terminal only):
```powershell
$env:ANTHROPIC_API_KEY="YOUR_API_KEY"
```
Do not share or commit your key. The API requires valid account access and may incur charges. If the chosen model ID is unavailable to your account, change `claude-3-5-haiku-latest` in `app.py` to a model available to you.

## Optional Google Sheets setup
1. In Google Cloud, enable Google Sheets API and create a service account.
2. Download the service-account JSON and save it as `service_account.json` in this folder (do not commit it).
3. Create a Google Sheet and share it with the service account's email address as Editor.
4. Set the spreadsheet ID in PowerShell:
   ```powershell
   $env:GOOGLE_SHEETS_SPREADSHEET_ID="YOUR_SPREADSHEET_ID"
   ```
5. The app writes only after the user gives consent and clicks the save button.

## Security notes
- `.gitignore` excludes `service_account.json`, local secrets, and model artifacts.
- Do not put API keys in source code or public repositories.
- Only send profile data to external services after informed consent.

## Dataset limitation
`data/loan_dataset.csv` is synthetic demonstration data generated for this project. It is not real lending data and must not be used for actual loan decisions. The accuracy is a software demonstration metric only.
