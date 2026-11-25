
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


# --------------------------------------------------
# 1. LOAD CLEANED DATA
# --------------------------------------------------

airbnb = pd.read_excel("data/my_processed/airbnb_final.xlsx")  # CHANGED

# Count listings before removing missing bedroom/bed information
listings_before = len(airbnb)

# CHANGED: no dropna. The 965 listings with missing bedrooms/beds are kept
# and filled inside the pipeline (see section 7). Deletion would remove 6%
# of the data, and mostly studios, hotels and boats, so not at random.
# CHANGED: listings with price 0 are removed instead (15 hotels, log(0) impossible)
airbnb = airbnb[airbnb["price"] > 0].copy()

listings_after = len(airbnb)
listings_removed = listings_before - listings_after

# print("Listings before removal:", listings_before)
# print("Listings after removal:", listings_after)
# print("Listings removed:", listings_removed)


# --------------------------------------------------
# 2. IDENTIFY AMENITY COLUMNS USING THE MAPPING
# --------------------------------------------------

amenity_mapping = pd.read_excel("data/amenityMapping.xlsx")

mapped_amenities = (
    amenity_mapping["new_proposed_amenity"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

# Only keep mapped amenities that exist in the Airbnb dataset
amenity_columns = [
    col for col in mapped_amenities
    if col in airbnb.columns
]

# Check that the identified amenity columns are binary
for col in amenity_columns:
    values = set(airbnb[col].dropna().unique())
    if not values.issubset({0, 1}):
        raise ValueError(
            f"Amenity column '{col}' contains values other than 0 and 1."
        )

# print("Mapped amenities found in dataset:", len(amenity_columns))


# --------------------------------------------------
# 3. FILTER AMENITIES BY PREVALENCE
# --------------------------------------------------

# Count how many listings have each amenity
amenity_counts = airbnb[amenity_columns].sum().sort_values(
    ascending=False
)

minimum_listings = 10

# Amenities included in the model
selected_amenities = amenity_counts[
    amenity_counts >= minimum_listings
].index.tolist()

# Rare amenities excluded from the model
rare_amenities = amenity_counts[
    amenity_counts < minimum_listings
].sort_values()

# print("Minimum listings required:", minimum_listings)
# print("Amenities included:", len(selected_amenities))
# print("Rare amenities excluded:", len(rare_amenities))


# --------------------------------------------------
# 4. DEFINE CANDIDATE FEATURES
# --------------------------------------------------

target = "price"

# Columns that should never be used as predictors
exclude_columns = [
    "id",
    "host_id",
    "price",
    "listing_url",
    "host_url",
    "host_name",
    "name",
    "description",
    "neighborhood_overview",
    "picture_url",
    "host_thumbnail_url",
    "host_picture_url",
    "amenities",
    "scrape_id",
    "last_scraped",
    "calendar_last_scraped",
    "first_review",
    "last_review",
    "host_since",
    "license",
    "review_scores_checkin",
    "review_scores_location",
    "review_scores_communication",
    "review_scores_accuracy",
    "review_scores_rating",
    # CHANGED: review_scores_cleanliness, longitude and latitude are no longer
    # excluded. Location is one of the strongest price drivers.
    # CHANGED: these are a RESULT of the price, not a cause of it. A cheap
    # listing gets booked more, so it has more reviews and less availability.
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "reviews_per_month",
    "review_scores_value",
    "host_listings_count",
    "host_total_listings_count"
]

# Categorical variables to evaluate
categorical_features = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "host_is_superhost"
    
]

# Keep only categorical columns that exist
categorical_features = [
    col for col in categorical_features
    if col in airbnb.columns
    and col not in exclude_columns
]

# Automatically identify remaining numerical columns
# Explicitly exclude all amenities to prevent duplication
numeric_features = [
    col for col in airbnb.select_dtypes(include=np.number).columns
    if col not in exclude_columns
    and col not in amenity_columns
]

# Combine all candidate features
feature_columns = (
    numeric_features
    + categorical_features
    + selected_amenities
)

X = airbnb[feature_columns].copy()
# CHANGED: the model learns log(price); np.exp() turns it back into euros
price = airbnb[target].to_numpy(dtype=float)
y = np.log(price)

print("\nFEATURE SELECTION")
print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("Amenity features:", len(selected_amenities))
print("Total candidate features:", len(feature_columns))


# --------------------------------------------------
# 5. CHECK MISSING NUMERICAL VALUES
# --------------------------------------------------

numeric_missing = X[numeric_features].isna().sum()
numeric_missing = numeric_missing[numeric_missing > 0]

# CHANGED: we no longer stop here. The gaps are filled in the pipeline,
# learning the median from the training rows only.
if not numeric_missing.empty:
    print("\nMissing values (filled with the training median in the pipeline):")
    print(numeric_missing.sort_values(ascending=False).to_string())


# --------------------------------------------------
# 6. TRAIN-TEST SPLIT
# --------------------------------------------------

# CHANGED: price goes through the split too, so we can measure the error in euros
X_train, X_test, y_train, y_test, price_train, price_test = train_test_split(
    X,
    y,
    price,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 7. PREPROCESSING
# --------------------------------------------------

# CHANGED: median imputation instead of "passthrough", so missing values
# no longer force us to delete rows
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("amenities", "passthrough", selected_amenities)
])


# --------------------------------------------------
# 8. RANDOM FOREST MODEL
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", RandomForestRegressor(
        n_estimators=600, # Number of trees in the forest
        random_state=42, # Random seed for reproducibility
        max_depth=20, # Maximum depth of each tree
        n_jobs=1,
        min_samples_leaf=10 # Minimum number of samples required to be at a leaf node
        
    ))
])




# --------------------------------------------------
# 9. TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nRandom Forest trained successfully.")


# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------

# CHANGED: the model predicts log(price), so convert back before measuring
y_pred = np.exp(model.predict(X_test))

mae = mean_absolute_error(price_test, y_pred)
rmse = np.sqrt(mean_squared_error(price_test, y_pred))
r2 = r2_score(price_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"MAE:  €{mae:.2f}")
print(f"RMSE: €{rmse:.2f}")
print(f"R²:   {r2:.3f}")


# --------------------------------------------------
# 11. PERMUTATION FEATURE IMPORTANCE
# --------------------------------------------------

importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5, # Number of times to permute a feature
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance": importance.importances_mean,
    "std": importance.importances_std
}).sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 10 FEATURES")
print(feature_importance.head(10).to_string(index=False))


# --------------------------------------------------
# 12. EXPORT RESULTS
# --------------------------------------------------

# feature_importance.to_excel(
#     "data/processed/feature_importance.xlsx",
#     index=False
# )

# # Export rare amenities separately for later analysis
# rare_amenities.to_frame(name="number_of_listings").to_excel(
#     "data/processed/rare_amenities.xlsx"
# )

# # Export all amenity counts for further analysis
# amenity_counts.to_frame(name="number_of_listings").to_excel(
#     "data/processed/amenity_prevalence.xlsx"
# )

# print("\nResults exported successfully.")
