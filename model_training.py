# Import Libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score, mean_squared_error

# Load Dataset
df = pd.read_csv("dataset/cardekho_dataset.csv")

# Display Dataset Information
print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# Remove unwanted column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Separate Features and Target
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

print("\nFeatures:")
print(X.columns)

# Categorical and Numerical Columns
categorical_cols = [
    "car_name",
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type"
]

numerical_cols = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

# One Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Random Forest Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Create Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Model...")
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("---------------------------")
print("R2 Score :", round(r2, 4))
print("RMSE     :", round(rmse, 2))

# Save Model
with open("best_model.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("\nModel saved successfully as best_model.pkl")