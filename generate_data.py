
import numpy as np
import pandas as pd

# Make the results reproducible
np.random.seed(42)

# Number of records
N = 5000

# Generate environmental conditions
temperature = np.random.uniform(15, 40, N)
humidity = np.random.uniform(25, 100, N)
rainfall = np.random.exponential(12, N)
rainfall = np.clip(rainfall, 0, 100)
soil_moisture = np.random.uniform(15, 100, N)

# Generate leaf spot observations
leaf_spots = np.random.binomial(1, 0.35, N)

# Calculate an illustrative risk score
score = (
    0.25 * (temperature >= 30)
    + 0.30 * (humidity >= 75)
    + 0.20 * (rainfall >= 15)
    + 0.15 * (soil_moisture >= 65)
    + 0.35 * leaf_spots
)

# Add small variability to avoid perfectly fixed labels
score = score + np.random.normal(0, 0.12, N)

# Convert the score into risk categories
risk = np.where(
    score >= 0.65,
    "High",
    np.where(score >= 0.35, "Medium", "Low")
)

# Create the dataset
df = pd.DataFrame({
    "temperature": temperature.round(2),
    "humidity": humidity.round(2),
    "rainfall": rainfall.round(2),
    "soil_moisture": soil_moisture.round(2),
    "leaf_spots": leaf_spots,
    "risk": risk
})

# Save the dataset
df.to_csv("dataset.csv", index=False)

# Display dataset information
print("Dataset generated successfully!")
print("Total rows:", len(df))
print("\nFirst 5 rows:")
print(df.head())

print("\nRisk distribution:")
print(df["risk"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
