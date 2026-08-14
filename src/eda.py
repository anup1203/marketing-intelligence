import pandas as pd
import matplotlib.pyplot as plt


# 1. LOAD DATASET


df = pd.read_csv("data/marketing_data.csv")


# 2. BASIC DATA CHECK


print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())


print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)


print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

print(df.columns.tolist())


print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# 3. MISSING VALUES


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

# 4. DUPLICATE ROWS


print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(df.duplicated().sum())


# 5. SUMMARY STATISTICS


print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(df.describe())


# 6. PLATFORM PERFORMANCE


print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE BY PLATFORM")
print("=" * 60)

platform_performance = (
    df.groupby("Platform")["Performance_Score"]
    .mean()
    .sort_values(ascending=False)
)

print(platform_performance)



# 7. CONTENT TYPE PERFORMANCE


print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE BY CONTENT TYPE")
print("=" * 60)

content_performance = (
    df.groupby("Content_Type")["Performance_Score"]
    .mean()
    .sort_values(ascending=False)
)

print(content_performance)



# 8. INDUSTRY PERFORMANCE


print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE BY INDUSTRY")
print("=" * 60)

industry_performance = (
    df.groupby("Industry")["Performance_Score"]
    .mean()
    .sort_values(ascending=False)
)

print(industry_performance)


# 9. CONTENT TOPIC PERFORMANCE


print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE BY CONTENT TOPIC")
print("=" * 60)

topic_performance = (
    df.groupby("Content_Topic")["Performance_Score"]
    .mean()
    .sort_values(ascending=False)
)

print(topic_performance)


# 10. POSTING DAY PERFORMANCE


print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE BY POSTING DAY")
print("=" * 60)

day_performance = (
    df.groupby("Posting_Day")["Performance_Score"]
    .mean()
    .sort_values(ascending=False)
)

print(day_performance)



# 11. POSTING TIME PERFORMANCE

print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE BY POSTING TIME")
print("=" * 60)

time_performance = (
    df.groupby("Posting_Time")["Performance_Score"]
    .mean()
    .sort_values(ascending=False)
)

print(time_performance)



# 12. CORRELATION ANALYSIS


print("\n" + "=" * 60)
print("CORRELATION WITH PERFORMANCE SCORE")
print("=" * 60)

numeric_columns = [
    "Reach",
    "Impressions",
    "Likes",
    "Comments",
    "Shares",
    "Saves",
    "Video_Views",
    "Watch_Time",
    "Clicks",
    "Leads",
    "Ad_Spend",
    "Revenue",
    "Performance_Score"
]

correlation = (
    df[numeric_columns]
    .corr()["Performance_Score"]
    .sort_values(ascending=False)
)

print(correlation)


# 13. PLATFORM PERFORMANCE CHART


plt.figure(figsize=(8, 5))

platform_performance.plot(kind="bar")

plt.title("Average Performance Score by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Performance Score")
plt.xticks(rotation=45)

plt.tight_layout()

plt.show(block=False)
plt.pause(2)
plt.close()



# 14. CONTENT TYPE PERFORMANCE CHART


plt.figure(figsize=(8, 5))

content_performance.plot(kind="bar")

plt.title("Average Performance Score by Content Type")
plt.xlabel("Content Type")
plt.ylabel("Average Performance Score")
plt.xticks(rotation=45)

plt.tight_layout()

plt.show(block=False)
plt.pause(2)
plt.close()


# 15. INDUSTRY PERFORMANCE CHART


plt.figure(figsize=(8, 5))

industry_performance.plot(kind="bar")

plt.title("Average Performance Score by Industry")
plt.xlabel("Industry")
plt.ylabel("Average Performance Score")
plt.xticks(rotation=45)

plt.tight_layout()

plt.show(block=False)
plt.pause(2)
plt.close()



# 16. TOPIC PERFORMANCE CHART


plt.figure(figsize=(9, 5))

topic_performance.plot(kind="bar")

plt.title("Average Performance Score by Content Topic")
plt.xlabel("Content Topic")
plt.ylabel("Average Performance Score")
plt.xticks(rotation=45)

plt.tight_layout()

plt.show(block=False)
plt.pause(2)
plt.close()



# 17. POSTING DAY PERFORMANCE CHART


plt.figure(figsize=(8, 5))

day_performance.plot(kind="bar")

plt.title("Average Performance Score by Posting Day")
plt.xlabel("Posting Day")
plt.ylabel("Average Performance Score")
plt.xticks(rotation=45)

plt.tight_layout()

plt.show(block=False)
plt.pause(2)
plt.close()



# 18. POSTING TIME PERFORMANCE CHART

plt.figure(figsize=(8, 5))

time_performance.plot(kind="bar")

plt.title("Average Performance Score by Posting Time")
plt.xlabel("Posting Time")
plt.ylabel("Average Performance Score")
plt.xticks(rotation=45)

plt.tight_layout()

plt.show(block=False)
plt.pause(2)
plt.close()



print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nDataset is ready for the next stage.")