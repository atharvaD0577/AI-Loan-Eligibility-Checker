from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_dataset.csv"
MODEL_PATH = ROOT / "models" / "loan_model.joblib"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"].map({"Y": 1, "N": 0})

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

prep = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocess", prep),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("=== AI LOAN ELIGIBILITY CHECKER ===")
print(f"Dataset rows: {len(df)}")
print(f"Test accuracy: {accuracy_score(y_test, pred):.2%}")
print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "features": X.columns.tolist()}, MODEL_PATH)
print(f"\nSaved model to: {MODEL_PATH}")
