
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# 1. Load dataset
airbnb = pd.read_excel(
    "data/processed/airbnb_final.xlsx"
)

# 2. Define target
y = airbnb["price"]

listings_detailed = pd.read_excel(
    "data/processed/listings_detailed_cleaned.xlsx"
)

# 3. Identify amenity columns
amenity_columns = [
    col for col in airbnb.columns
    if col not in listings_detailed.columns
    and col != "id"
]

# 4. Define other predictors
numeric_features = [
    "accommodates",
    "bedrooms",
    "beds",
    "latitude",
    "longitude"
]

categorical_features = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed"
]

# 5. Combine all predictors
feature_columns = (
    numeric_features
    + categorical_features
    + amenity_columns
)

X = airbnb[feature_columns]

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("amenities", "passthrough", amenity_columns)
])

# 8. Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5
    ))
])

# 9. Train model
model.fit(X_train, y_train)

print("Random Forest trained successfully.")
