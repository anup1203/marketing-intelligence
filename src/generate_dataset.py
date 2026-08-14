import pandas as pd
import numpy as np
import random

# Number of records
num_records = 500

# Possible values
industries = [
    "Fashion",
    "Food",
    "Technology",
    "Finance",
    "Education",
    "Healthcare"
]

platforms = [
    "Instagram",
    "YouTube",
    "Facebook",
    "LinkedIn",
    "TikTok"
]

content_types = [
    "Reel",
    "Video",
    "Image",
    "Carousel",
    "Article"
]

topics = [
    "Product Education",
    "Tutorial",
    "Promotion",
    "Entertainment",
    "Tips",
    "Customer Story",
    "Behind the Scenes"
]

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

posting_times = [
    "09:00",
    "11:00",
    "13:00",
    "15:00",
    "17:00",
    "19:00",
    "21:00"
]

data = []

for i in range(num_records):

    reach = random.randint(1000, 100000)
    impressions = random.randint(reach, reach * 2)

    likes = random.randint(50, max(100, int(reach * 0.15)))
    comments = random.randint(5, max(10, int(likes * 0.15)))
    shares = random.randint(5, max(10, int(likes * 0.25)))
    saves = random.randint(5, max(10, int(likes * 0.30)))

    video_views = random.randint(0, reach)

    watch_time = random.randint(
        0,
        max(100, video_views * 5)
    )

    clicks = random.randint(
        10,
        max(20, int(reach * 0.10))
    )

    leads = random.randint(
        0,
        max(5, int(clicks * 0.20))
    )

    ad_spend = round(random.uniform(500, 20000), 2)

    revenue = round(
        ad_spend * random.uniform(0.8, 8.0),
        2
    )

    # Create a performance score from multiple engagement signals
    engagement_rate = (
        (likes + comments + shares + saves)
        / max(reach, 1)
    )

    click_rate = clicks / max(impressions, 1)

    lead_rate = leads / max(clicks, 1)

    roi = revenue / max(ad_spend, 1)

    engagement_score = 35 * (
        1 - np.exp(-engagement_rate / 0.05)
    )

    click_score = 25 * (
        1 - np.exp(-click_rate / 0.03)
    )

    lead_score = 20 * (
        1 - np.exp(-lead_rate / 0.08)
    )

    roi_score = 20 * (
        1 - np.exp(-roi / 4)
    )

    performance_score = round(
        min(99.5, engagement_score + click_score + lead_score + roi_score),
        2
    )

    row = {
        "Client": f"Client_{random.randint(1, 20)}",
        "Industry": random.choice(industries),
        "Platform": random.choice(platforms),
        "Content_Type": random.choice(content_types),
        "Content_Topic": random.choice(topics),
        "Posting_Day": random.choice(days),
        "Posting_Time": random.choice(posting_times),

        "Reach": reach,
        "Impressions": impressions,
        "Likes": likes,
        "Comments": comments,
        "Shares": shares,
        "Saves": saves,
        "Video_Views": video_views,
        "Watch_Time": watch_time,
        "Clicks": clicks,
        "Leads": leads,
        "Ad_Spend": ad_spend,
        "Revenue": revenue,

        "Performance_Score": performance_score
    }

    data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Basic validation
print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nSummary:")
print(df.describe())

# Save CSV
df.to_csv("data/marketing_data.csv", index=False)

print("\nDataset created successfully!")
print("File: data/marketing_data.csv")