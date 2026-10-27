# Practical 5 - Version 2: Decision Tree on Loan Approval
# Rename the loan CSV to loan__approval_data.csv.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("loan__approval_data.csv")
X = df.drop(columns=["Loan_Status", "Applicant_ID"], errors="ignore")
y = df["Loan_Status"]

numeric_cols = X.select_dtypes(include="number").columns
text_cols = X.select_dtypes(exclude="number").columns

prep = ColumnTransformer([
    ("numbers", SimpleImputer(strategy="median"), numeric_cols),
    ("text", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), text_cols)
])

model = Pipeline([
    ("prepare", prep),
    ("tree", DecisionTreeClassifier(max_depth=5, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
