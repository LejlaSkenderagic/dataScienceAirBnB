import pandas as pd 
import ast #It safely interprets strings containing Python literals, including lists, without executing arbitrary code.
from sklearn.preprocessing import MultiLabelBinarizer
import os
import warnings


OUTPUT = "data/my_processed"
os.makedirs(OUTPUT, exist_ok=True)
warnings.filterwarnings("ignore", message="Ignoring URL")

# NEW: the date this data was scraped (column last_scraped)
SCRAPE_DATE = pd.Timestamp("2021-09-07")

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

# CHANGED: takes the table as an argument and returns it, so the fix
# reaches the exported files. In the team version it changes a global
# and is called after the exports, so its 503 corrections are lost.
def missing_values_bedroom(listings_detailed):
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
    listings_detailed.loc[confirmed_studio_candidates, "bedrooms"] = 1

    # NEW: studios are cheaper than real 1-bedroom flats (median EUR 100
    # vs EUR 122 for entire homes for 2 guests), so mark them
    listings_detailed["is_studio"] = confirmed_studio_candidates.astype(int)

    # NEW: export the studio listings themselves
    listings_detailed.loc[
        confirmed_studio_candidates, ["id", "name", "bedrooms"]
    ].to_excel(f"{OUTPUT}/studio_candidates.xlsx", index=False)

    print("Listings classified as studios:",
    confirmed_studio_candidates.sum())

    print("Bedrooms still missing:",
    listings_detailed["bedrooms"].isna().sum())

    return listings_detailed

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
    mapping = pd.read_excel("data/amenityMapping.xlsx")

    # 2. Create a dictionary connecting original names to standardised names
    amenity_dict = dict(zip(
        mapping["original_amenity"],
        mapping["new_proposed_amenity"]
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

# NEW: extra columns built from columns we already have. Each one reads
# only values from its own row, so by the tutorial's rule they belong
# before the split.
def add_extra_features(df):
    # the dataset has no numeric bathrooms column, only text like
    # "1.5 shared baths" or "1 private bath"
    bathrooms_text = df["bathrooms_text"].astype("string").str.lower()
    df["bathrooms"] = pd.to_numeric(
        bathrooms_text.str.extract(r"([\d.]+)")[0],
        errors="coerce"
    )
    half_bath = bathrooms_text.str.contains("half", na=False) & df["bathrooms"].isna()
    df.loc[half_bath, "bathrooms"] = 0.5
    df["bathroom_shared"] = bathrooms_text.str.contains("shared", na=False).astype(int)

    # how many years the host has been hosting
    df["host_years"] = (
        SCRAPE_DATE - pd.to_datetime(df["host_since"], errors="coerce")
    ).dt.days / 365

    # how many years the listing has been active
    df["listing_years"] = (
        SCRAPE_DATE - pd.to_datetime(df["first_review"], errors="coerce")
    ).dt.days / 365

    # 1 if the name or description mentions luxury or design
    text = (df["name"].fillna("") + " " + df["description"].fillna("")).str.lower()
    df["says_luxury"] = text.str.contains(r"luxur|design", regex=True).astype(int)

    return df

# CHANGED: the studio fix now runs BEFORE the exports
listings_detailed = missing_values_bedroom(listings_detailed)

listings_detailed = clean_price(listings_detailed)

# NEW
listings_detailed = add_extra_features(listings_detailed)

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
    f"{OUTPUT}/amenity_binary_table_standardised.xlsx",
    index=False
)


# Export cleaned datasets
listings.to_excel(
    f"{OUTPUT}/listings_cleaned.xlsx",
    index=False
)

listings_detailed.to_excel(
    f"{OUTPUT}/listings_detailed_cleaned.xlsx",
    index=False
)

# Export standardised binary amenity table
standardised_table.to_excel(
    f"{OUTPUT}/amenity_binary_table_standardised.xlsx",
    index=False
)

# Export merged dataset for analysis
airbnb_final.to_excel(
    f"{OUTPUT}/airbnb_final.xlsx",
    index=False
)


print("Final dataset:", airbnb_final.shape[0], "rows,",
airbnb_final.shape[1], "columns")
print(f"Saved to {OUTPUT}/ -- the team's data/processed/ was not touched.")
