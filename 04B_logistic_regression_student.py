# Practical 4 - Version 2: Logistic Regression using Student data
# IMPORTANT: Replace these example values with the exact arrays in your practical sheet.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    "Study_Hours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    "Attendance":  [50, 55, 60, 60, 65, 70, 75, 75, 85, 90],
    "Pass":        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})

X = data[["Study_Hours", "Attendance"]]
y = data["Pass"]

model = LogisticRegression()
model.fit(X, y)
pred = model.predict(X)

print("Predictions (0 = Fail, 1 = Pass):", pred)
print("Accuracy on these examples:", accuracy_score(y, pred))
print("Confusion matrix:\n", confusion_matrix(y, pred))
print("Prediction for 4 study hours and 75 attendance:",
      model.predict(pd.DataFrame([[4, 75]], columns=X.columns))[0])
