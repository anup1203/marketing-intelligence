import pandas as pd

# 1. LOAD DATASET

df = pd.read_csv("data/marketing_data.csv")

# 2. CREATE MARKETING FEATURES


# Engagement Rate
df["Engagement_Rate"] = (
    (df["Likes"] + df["Comments"] + df["Shares"] + df["Saves"])
    / df["Reach"].replace(0, 1)
) * 100


# Click Through Rate (CTR)
df["CTR"] = (
    df["Clicks"]
    / df["Impressions"].replace(0, 1)
) * 100


# Lead Conversion Rate
df["Lead_Conversion_Rate"] = (
    df["Leads"]
    / df["Clicks"].replace(0, 1)
) * 100


# Return on Investment (ROI)
df["ROI"] = (
    (df["Revenue"] - df["Ad_Spend"])
    / df["Ad_Spend"].replace(0, 1)
) * 100


# Return on Ad Spend (ROAS)
df["ROAS"] = (
    df["Revenue"]
    / df["Ad_Spend"].replace(0, 1)
)

# 3. ROUND NEW FEATURES

df["Engagement_Rate"] = df["Engagement_Rate"].round(2)
df["CTR"] = df["CTR"].round(2)
df["Lead_Conversion_Rate"] = df["Lead_Conversion_Rate"].round(2)
df["ROI"] = df["ROI"].round(2)
df["ROAS"] = df["ROAS"].round(2)

# 4. CHECK NEW DATASET

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nNew columns:")
print([
    "Engagement_Rate",
    "CTR",
    "Lead_Conversion_Rate",
    "ROI",
    "ROAS"
])


print("\nFirst 5 records:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nMissing Values:")
print(df.isnull().sum())


# 5. SAVE FEATURE-ENGINEERED DATASET

df.to_csv("data/marketing_data_features.csv", index=False)


print("\n" + "=" * 60)
print("Feature-engineered dataset saved successfully!")
print("File: data/marketing_data_features.csv")
print("=" * 60)
