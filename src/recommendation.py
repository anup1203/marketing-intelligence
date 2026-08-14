import pandas as pd


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATASET
# ============================================================

df = pd.read_csv("data/marketing_data_features.csv")


# ============================================================
# 2. SHOW AVAILABLE OPTIONS
# ============================================================

print("\n" + "=" * 60)
print("MARKETING RECOMMENDATION ENGINE")
print("=" * 60)

print("\nAvailable Industries:")
print(df["Industry"].unique())

print("\nAvailable Platforms:")
print(df["Platform"].unique())


# ============================================================
# 3. USER INPUT
# ============================================================

industry = input("\nEnter Industry: ").strip()
platform = input("Enter Platform: ").strip()


# ============================================================
# 4. FILTER DATA
# ============================================================

filtered_df = df[
    (df["Industry"].str.lower() == industry.lower()) &
    (df["Platform"].str.lower() == platform.lower())
]


# ============================================================
# 5. CHECK IF DATA EXISTS
# ============================================================

if filtered_df.empty:

    print("\n" + "=" * 60)
    print("NO MATCHING DATA FOUND")
    print("=" * 60)

    print("\nPlease enter an Industry and Platform from the available options.")

else:

    # ========================================================
    # 6. FIND BEST CONTENT TYPE
    # ========================================================

    best_content = (
        filtered_df.groupby("Content_Type")
        .agg(
            Performance_Score=("Performance_Score", "mean"),
            Engagement_Rate=("Engagement_Rate", "mean"),
            CTR=("CTR", "mean"),
            ROAS=("ROAS", "mean")
        )
        .sort_values("Performance_Score", ascending=False)
    )

    best_content_type = best_content.index[0]


    # ========================================================
    # 7. FIND BEST CONTENT TOPIC
    # ========================================================

    best_topic = (
        filtered_df.groupby("Content_Topic")
        .agg(
            Performance_Score=("Performance_Score", "mean"),
            Engagement_Rate=("Engagement_Rate", "mean"),
            CTR=("CTR", "mean"),
            ROAS=("ROAS", "mean")
        )
        .sort_values("Performance_Score", ascending=False)
    )

    best_content_topic = best_topic.index[0]


    # ========================================================
    # 8. FIND BEST POSTING DAY
    # ========================================================

    best_day = (
        filtered_df.groupby("Posting_Day")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    best_posting_day = best_day.index[0]


    # ========================================================
    # 9. FIND BEST POSTING TIME
    # ========================================================

    best_time = (
        filtered_df.groupby("Posting_Time")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    best_posting_time = best_time.index[0]


    # ========================================================
    # 10. GET EXPECTED PERFORMANCE METRICS
    # ========================================================

    recommended_data = filtered_df[
        (filtered_df["Content_Type"] == best_content_type) &
        (filtered_df["Content_Topic"] == best_content_topic) &
        (filtered_df["Posting_Day"] == best_posting_day) &
        (filtered_df["Posting_Time"] == best_posting_time)
    ]


    # If exact combination does not exist,
    # use all filtered records for average metrics.

    if recommended_data.empty:
        recommended_data = filtered_df


    expected_performance = recommended_data["Performance_Score"].mean()
    expected_engagement = recommended_data["Engagement_Rate"].mean()
    expected_ctr = recommended_data["CTR"].mean()
    expected_roas = recommended_data["ROAS"].mean()


    # ========================================================
    # 11. DISPLAY RECOMMENDATION
    # ========================================================

    print("\n" + "=" * 60)
    print("RECOMMENDED MARKETING STRATEGY")
    print("=" * 60)

    print(f"\nIndustry        : {industry}")
    print(f"Platform        : {platform}")

    print(f"\nRecommended Content Type : {best_content_type}")
    print(f"Recommended Topic       : {best_content_topic}")
    print(f"Recommended Posting Day : {best_posting_day}")
    print(f"Recommended Posting Time: {best_posting_time}")

    print("\nExpected Metrics:")
    print(f"Performance Score : {expected_performance:.2f}")
    print(f"Engagement Rate   : {expected_engagement:.2f}%")
    print(f"CTR               : {expected_ctr:.2f}%")
    print(f"ROAS              : {expected_roas:.2f}")


    # ========================================================
    # 12. SHOW TOP CONTENT TYPES
    # ========================================================

    print("\n" + "=" * 60)
    print("TOP CONTENT TYPES FOR THIS SELECTION")
    print("=" * 60)

    print(best_content.round(2))


    # ========================================================
    # 13. SHOW TOP CONTENT TOPICS
    # ========================================================

    print("\n" + "=" * 60)
    print("TOP CONTENT TOPICS FOR THIS SELECTION")
    print("=" * 60)

    print(best_topic.round(2))


    # ========================================================
    # 14. FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 60)
    print("RECOMMENDATION GENERATED SUCCESSFULLY!")
    print("=" * 60)