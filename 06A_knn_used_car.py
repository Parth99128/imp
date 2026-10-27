# Practical 6 - Version 1: KNN Regression on Used Car Dataset
# Rename the dataset to used_car_dataset.csv.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("used_car_dataset.csv")

# Convert price strings such as '₹ 4.45 Lakh' to rupees.
def price_to_number(value):
    value = str(value).replace("₹", "").replace(",", "").strip()
    if "Lakh" in value:
        return float(value.replace("Lakh", "").strip()) * 100000
    if "Crore" in value:
        return float(value.replace("Crore", "").strip()) * 10000000
    return float(value)

# Convert mileage strings such as '22,402 km' to numbers.
def km_to_number(value):
    return float(str(value).lower().replace("km", "").replace(",", "").strip())

df["price_num"] = df["car_price_in_rupees"].apply(price_to_number)
df["kms_num"] = df["kms_driven"].apply(km_to_number)

X = df[["kms_num", "fuel_type", "city", "year_of_manufacture"]]
y = df["price_num"]

prep = ColumnTransformer([
    ("numbers", Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), ["kms_num", "year_of_manufacture"]),
    ("categories", OneHotEncoder(handle_unknown="ignore"), ["fuel_type", "city"])
])

model = Pipeline([
    ("prepare", prep),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2 score:", r2_score(y_test, pred))
