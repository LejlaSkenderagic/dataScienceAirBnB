import pandas as pd 
import ast #It safely interprets strings containing Python literals, including lists, without executing arbitrary code.
from sklearn.preprocessing import MultiLabelBinarizer
import os

airbnb_data = pd.read_excel(
    "data/raw/Airbnb_Amsterdam_2022_combined.xlsx",
    sheet_name=None                                             #sheet_name=None means all sheets will be read if =certain sheet then only that one will be read
)

# Access each individual dataset
listings = airbnb_data["listings"]
listings_detailed = airbnb_data["listings_detailed"]
reviews = airbnb_data["reviews"]
calendar_summary = airbnb_data["calendar_summary"]
neighbourhoods = airbnb_data["neighbourhoods"]


def remove_empty_columns(data):
    for name, df in data.items():

        # Identify columns that are completely empty
        empty_columns = df.columns[df.isna().all()].tolist()

        # Remove those columns
        data[name] = df.drop(columns=empty_columns)

        # Print the results
        # print(f"{name}: removed columns: {empty_columns}")

def missing_values_bedroom(data):
    # Identify listings where bedrooms are missing
    missing_bedrooms = listings_detailed["bedrooms"].isna()

    # Check whether the description mentions "studio"
    studio_mentioned = (
    listings_detailed["description"]
    .fillna("")
    .str.contains(r"\bstudio\b", case=False, regex=True)
    )

    # Only classify listings where bedrooms are missing AND studio is mentioned
    confirmed_studio_candidates = missing_bedrooms & studio_mentioned

    # Assign zero bedrooms to these listings
    listings_detailed.loc[confirmed_studio_candidates, "bedrooms"] = 0

    # # Print results
    # print("Listings classified as studios:",
    # confirmed_studio_candidates.sum())

    confirmed_studio_candidates.to_excel(
        "data/processed/studio_candidates.xlsx",
        index=False
    )

    # print("Bedrooms still missing:",
    # listings_detailed["bedrooms"].isna().sum())

remove_empty_columns(airbnb_data)
# Access each individual dataset
listings = airbnb_data["listings"]
listings_detailed = airbnb_data["listings_detailed"]
reviews = airbnb_data["reviews"]
calendar_summary = airbnb_data["calendar_summary"]
neighbourhoods = airbnb_data["neighbourhoods"]

def create_amenity_table(listings_detailed):

    # 1. Convert amenities into Python lists
    listings_detailed["amenities"] = listings_detailed["amenities"].apply(ast.literal_eval)

    # 2. Create binary amenity table
    mlb = MultiLabelBinarizer()

    amenity_matrix = mlb.fit_transform(
        listings_detailed["amenities"]
    )

    amenity_table = pd.DataFrame(
        amenity_matrix,
        columns=mlb.classes_,
        index=listings_detailed.index
    )

    # 3. Attach Airbnb listing IDs
    amenity_table.insert(
        0,
        "id",
        listings_detailed["id"].values
    )

    return amenity_table

def standardise_amenities(amenity_table):

    # 1. Load the mapping file
    mapping = pd.read_csv("data/amenity_mapping.csv")

    # 2. Create a dictionary connecting original names to standardised names
    amenity_dict = dict(zip(
        mapping["original_amenity"],
        mapping["standardised_amenity"]
    ))

    # 3. Separate listing IDs from amenity columns
    amenity_columns = amenity_table.drop(columns=["id"])

    # 4. Check whether every amenity has a mapping
    unmapped = [
        amenity for amenity in amenity_columns.columns
        if amenity not in amenity_dict
    ]

    if unmapped:
        raise ValueError(f"These amenities have no mapping: {unmapped}")

    # 5. Rename the original amenity columns
    standardised_table = amenity_columns.rename(
        columns=amenity_dict
    )

    # 6. Merge columns with identical standardised names
    standardised_table = (
        standardised_table.T
        .groupby(level=0)
        .max()
        .T
    )

    # 7. Add the listing IDs back
    standardised_table.insert(
        0,
        "id",
        amenity_table["id"].values
    )

    return standardised_table

def clean_price(df):
    df["price"] = (
        df["price"]
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    return df
listings_detailed = clean_price(listings_detailed)

# Create the original binary amenity table
amenity_table = create_amenity_table(listings_detailed)

# Standardise the amenity names
standardised_table = standardise_amenities(amenity_table)

airbnb_final = listings_detailed.merge(
    standardised_table,
    on="id",
    how="left",
    validate="one_to_one"
)

# Calculate amenity frequency
amenity_frequency = (
    standardised_table.drop(columns=["id"])
    .sum()
    .sort_values(ascending=False)
)

# Export the standardised binary table
standardised_table.to_excel(
    "data/processed/amenity_binary_table_standardised.xlsx",
    index=False
)


# Create processed folder if it does not exist
os.makedirs("data/processed", exist_ok=True)

# Export cleaned datasets
listings.to_excel(
    "data/processed/listings_cleaned.xlsx",
    index=False
)

listings_detailed.to_excel(
    "data/processed/listings_detailed_cleaned.xlsx",
    index=False
)

# Export standardised binary amenity table
standardised_table.to_excel(
    "data/processed/amenity_binary_table_standardised.xlsx",
    index=False
)

# Export merged dataset for analysis
airbnb_final.to_excel(
    "data/processed/airbnb_final.xlsx",
    index=False
)


# 1. Load data
airbnb_data = pd.read_excel(
    "data/raw/Airbnb_Amsterdam_2022_combined.xlsx",
    sheet_name=None
)

# 2. Remove empty columns
remove_empty_columns(airbnb_data)

# 3. Access cleaned datasets
listings = airbnb_data["listings"]
listings_detailed = airbnb_data["listings_detailed"]

# 4. Identify possible studios
studio_candidates = missing_values_bedroom(listings_detailed)


# 5. Clean price
listings_detailed = clean_price(listings_detailed)

# 6. Create binary amenity table
amenity_table = create_amenity_table(listings_detailed)

# 7. Standardise amenities
standardised_table = standardise_amenities(amenity_table)

# 8. Merge datasets
airbnb_final = listings_detailed.merge(
    standardised_table,
    on="id",
    how="left",
    validate="one_to_one"
)

# def inspect_missing_values(data):
#     for name, df in data.items():
#         print(listings_detailed["bedrooms"].value_counts(dropna=False))

# inspect_missing_values(airbnb_data)

# Find listings with missing bedroom information
# missing_bedrooms = listings_detailed[
#     listings_detailed["bedrooms"].isna()
# ]

# Display their listing IDs
# print(missing_bedrooms["id"].to_list())

# Show how many listings are affected
# print("Listings with missing bedrooms:", len(missing_bedrooms))

# Display results
# print(standardised_table.head())

# print("Original amenities:", amenity_table.shape[1] - 1)
# print("Standardised amenities:", standardised_table.shape[1] - 1)
# print("Total listings:", len(standardised_table))


# missing_values_bedroom(airbnb_data)

# clean_price(airbnb_data)

# print(listings_detailed[
#     ["price", "bedrooms", "beds", "accommodates",
#      "latitude", "longitude", "minimum_nights"]
# ].dtypes)



