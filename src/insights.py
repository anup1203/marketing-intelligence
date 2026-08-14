import pandas as pd


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATASET
# ============================================================

df = pd.read_csv("data/marketing_data_features.csv")


print("\n" + "=" * 60)
print("MARKETING INSIGHTS ANALYSIS")
print("=" * 60)


# ============================================================
# 2. BEST PLATFORM
# ============================================================

platform = (
    df.groupby("Platform")
    .agg(
        Performance_Score=("Performance_Score", "mean"),
        Engagement_Rate=("Engagement_Rate", "mean"),
        CTR=("CTR", "mean"),
        ROAS=("ROAS", "mean")
    )
    .sort_values("Performance_Score", ascending=False)
)

print("\nBEST PLATFORMS")
print(platform.round(2))


# ============================================================
# 3. BEST CONTENT TYPE
# ============================================================

content_type = (
    df.groupby("Content_Type")
    .agg(
        Performance_Score=("Performance_Score", "mean"),
        Engagement_Rate=("Engagement_Rate", "mean"),
        CTR=("CTR", "mean"),
        ROAS=("ROAS", "mean")
    )
    .sort_values("Performance_Score", ascending=False)
)

print("\nBEST CONTENT TYPES")
print(content_type.round(2))


# ============================================================
# 4. BEST INDUSTRY
# ============================================================

industry = (
    df.groupby("Industry")
    .agg(
        Performance_Score=("Performance_Score", "mean"),
        Engagement_Rate=("Engagement_Rate", "mean"),
        CTR=("CTR", "mean"),
        ROAS=("ROAS", "mean")
    )
    .sort_values("Performance_Score", ascending=False)
)

print("\nBEST INDUSTRIES")
print(industry.round(2))


# ============================================================
# 5. BEST CONTENT TOPIC
# ============================================================

topic = (
    df.groupby("Content_Topic")
    .agg(
        Performance_Score=("Performance_Score", "mean"),
        Engagement_Rate=("Engagement_Rate", "mean"),
        CTR=("CTR", "mean"),
        ROAS=("ROAS", "mean")
    )
    .sort_values("Performance_Score", ascending=False)
)

print("\nBEST CONTENT TOPICS")
print(topic.round(2))


# ============================================================
# 6. BEST POSTING DAY
# ============================================================

day = (
    df.groupby("Posting_Day")
    .agg(
        Performance_Score=("Performance_Score", "mean"),
        Engagement_Rate=("Engagement_Rate", "mean"),
        CTR=("CTR", "mean"),
        ROAS=("ROAS", "mean")
    )
    .sort_values("Performance_Score", ascending=False)
)

print("\nBEST POSTING DAYS")
print(day.round(2))


# ============================================================
# 7. BEST POSTING TIME
# ============================================================

time = (
    df.groupby("Posting_Time")
    .agg(
        Performance_Score=("Performance_Score", "mean"),
        Engagement_Rate=("Engagement_Rate", "mean"),
        CTR=("CTR", "mean"),
        ROAS=("ROAS", "mean")
    )
    .sort_values("Performance_Score", ascending=False)
)

print("\nBEST POSTING TIMES")
print(time.round(2))


# ============================================================
# 8. TOP ROI CAMPAIGNS
# ============================================================

top_roi = (
    df.sort_values("ROI", ascending=False)
    [
        [
            "Client",
            "Platform",
            "Content_Type",
            "Content_Topic",
            "Ad_Spend",
            "Revenue",
            "ROI",
            "ROAS",
            "Performance_Score"
        ]
    ]
    .head(10)
)

print("\nTOP 10 RECORDS BY ROI")
print(top_roi.round(2))


# ============================================================
# 9. TOP ROAS CAMPAIGNS
# ============================================================

top_roas = (
    df.sort_values("ROAS", ascending=False)
    [
        [
            "Client",
            "Platform",
            "Content_Type",
            "Content_Topic",
            "Ad_Spend",
            "Revenue",
            "ROI",
            "ROAS",
            "Performance_Score"
        ]
    ]
    .head(10)
)

print("\nTOP 10 RECORDS BY ROAS")
print(top_roas.round(2))


# ============================================================
# 10. TOP ENGAGEMENT
# ============================================================

top_engagement = (
    df.sort_values("Engagement_Rate", ascending=False)
    [
        [
            "Client",
            "Platform",
            "Content_Type",
            "Content_Topic",
            "Engagement_Rate",
            "Performance_Score"
        ]
    ]
    .head(10)
)

print("\nTOP 10 RECORDS BY ENGAGEMENT")
print(top_engagement.round(2))


# ============================================================
# 11. OVERALL BEST COMBINATION
# ============================================================

combination = (
    df.groupby(
        ["Platform", "Content_Type", "Content_Topic"]
    )
    .agg(
        Performance_Score=("Performance_Score", "mean"),
        Engagement_Rate=("Engagement_Rate", "mean"),
        CTR=("CTR", "mean"),
        ROAS=("ROAS", "mean"),
        Records=("Client", "count")
    )
    .sort_values("Performance_Score", ascending=False)
)

print("\nTOP MARKETING COMBINATIONS")
print(combination.head(10).round(2))


# ============================================================
# 12. FINAL KEY INSIGHTS
# ============================================================

best_platform = platform.index[0]
best_content = content_type.index[0]
best_industry = industry.index[0]
best_topic = topic.index[0]
best_day = day.index[0]
best_time = time.index[0]

print("\n" + "=" * 60)
print("KEY MARKETING INSIGHTS")
print("=" * 60)

print(f"Best Platform       : {best_platform}")
print(f"Best Content Type   : {best_content}")
print(f"Best Industry       : {best_industry}")
print(f"Best Content Topic  : {best_topic}")
print(f"Best Posting Day    : {best_day}")
print(f"Best Posting Time   : {best_time}")

print("\n" + "=" * 60)
print("INSIGHTS ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)