import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD DATASET

df = pd.read_csv("data/marketing_data_features.csv")

print("=" * 60)
print("MARKETING PERFORMANCE PREDICTION MODEL")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)
# 2. DEFINE TARGET

target = "Performance_Score"

X = df.drop(columns=[target])
y = df[target]

# 3. CATEGORICAL & NUMERICAL FEATURES

categorical_features = [
    "Client",
    "Industry",
    "Platform",
    "Content_Type",
    "Content_Topic",
    "Posting_Day",
    "Posting_Time"
]

numerical_features = [
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
    "Engagement_Rate",
    "CTR",
    "Lead_Conversion_Rate",
    "ROI",
    "ROAS"
]

# 4. PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# 5. MODEL

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
# 6. PIPELINE

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# 7. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 8. TRAIN MODEL

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")

# 9. PREDICTION
y_pred = pipeline.predict(X_test)


# 10. MODEL EVALUATION

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# 11. SAVE MODEL

model_path = "models/performance_model.pkl"

joblib.dump(pipeline, model_path)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY!")
print("=" * 60)

print(f"File: {model_path}")
