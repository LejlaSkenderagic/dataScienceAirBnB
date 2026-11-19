import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned Airbnb dataset
airbnb = pd.read_excel(
    "data/processed/airbnb_final.xlsx"
)

listings_detailed = pd.read_excel(
    "data/processed/listings_detailed_cleaned.xlsx"
)
listings = pd.read_excel(
    "data/processed/listings_cleaned.xlsx"
)

print("Dataset successfully loaded.")

# Dataset dimensions
print("\nDataset dimensions:")
print("Rows:", airbnb.shape[0])
print("Columns:", airbnb.shape[1])

# First observations
print("\nFirst five rows:")
print(airbnb.head())

# Column names and data types
print("\nDataset information:")
print(airbnb.info())

# Missing values
print("\nMissing values:")
print(
    airbnb.isna().sum()
    .sort_values(ascending=False)
    .head(30)
)

# Descriptive statistics for price
print("\nPrice statistics:")
print(airbnb["price"].describe())

# Median price
print("\nMedian price:", airbnb["price"].median())

# Price distribution
plt.figure(figsize=(10, 5))
sns.histplot(airbnb["price"].dropna(), bins=50)
plt.title("Distribution of Airbnb Listing Prices")
plt.xlabel("Price (€)")
plt.ylabel("Number of Listings")
plt.xlim(0, 1000) 
plt.show()

# Room type distribution
print("\nRoom types:")
print(airbnb["room_type"].value_counts())

# Property type distribution
print("\nTop 15 property types:")
print(airbnb["property_type"].value_counts().head(15))

# Accommodation capacity
print("\nAccommodation capacity:")
print(airbnb["accommodates"].describe())

# Bedrooms
print("\nBedrooms:")
print(airbnb["bedrooms"].value_counts(dropna=False).sort_index())

plt.figure(figsize=(8, 5))

sns.countplot(
    data=airbnb,
    x="room_type",
    order=airbnb["room_type"].value_counts().index
)

plt.title("Distribution of Airbnb Room Types")
plt.xlabel("Room Type")
plt.ylabel("Number of Listings")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()



# Identify amenity columns
amenity_columns = [
    col for col in airbnb.columns
    if col not in listings_detailed.columns
    and col != "id"
]

# Calculate frequency of each amenity
amenity_frequency = (
    airbnb[amenity_columns]
    .sum()
    .sort_values(ascending=False)
)

# Show 20 most common amenities
print("\n20 most common amenities:")
print(amenity_frequency.head(20))

# Visualise
plt.figure(figsize=(10, 7))

amenity_frequency.head(20).sort_values().plot(kind="barh")

plt.title("20 Most Common Airbnb Amenities")
plt.xlabel("Number of Listings")
plt.ylabel("Amenity")
plt.tight_layout()
plt.show()