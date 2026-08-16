import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/marketing_data_features.csv")

print("=" * 60)
print("MARKETING PERFORMANCE PREDICTION MODEL")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

target = "Performance_Score"

# Only information available before publishing
features = [
    "Industry",
    "Platform",
    "Content_Type",
    "Content_Topic",
    "Posting_Day",
    "Posting_Time"
]

X = df[features]
y = df[target]


# ============================================================
# 3. CATEGORICAL FEATURES
# ============================================================

categorical_features = features


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# ============================================================
# 7. BASELINE
# ============================================================

baseline_prediction = y_train.mean()

baseline_predictions = [baseline_prediction] * len(y_test)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = mean_squared_error(
    y_test,
    baseline_predictions
) ** 0.5

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


print("\n" + "=" * 60)
print("BASELINE MODEL")
print("=" * 60)

print(f"MAE  : {baseline_mae:.2f}")
print(f"RMSE : {baseline_rmse:.2f}")
print(f"R²   : {baseline_r2:.3f}")


# ============================================================
# 8. TRAIN AND COMPARE MODELS
# ============================================================

results = []

trained_pipelines = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )

    trained_pipelines[name] = pipeline


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 10. SAVE GRADIENT BOOSTING MODEL
# ============================================================

final_model = trained_pipelines["Gradient Boosting"]

model_path = "models/performance_model.pkl"

joblib.dump(
    final_model,
    model_path
)

print("\n" + "=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)

print(f"Model: Gradient Boosting")
print(f"File : {model_path}")


# ============================================================
# 11. FINAL MODEL TEST
# ============================================================

final_predictions = final_model.predict(X_test)

final_mae = mean_absolute_error(
    y_test,
    final_predictions
)

final_rmse = mean_squared_error(
    y_test,
    final_predictions
) ** 0.5

final_r2 = r2_score(
    y_test,
    final_predictions
)

print("\nFinal Model Performance:")
print(f"MAE  : {final_mae:.2f}")
print(f"RMSE : {final_rmse:.2f}")
print(f"R²   : {final_r2:.3f}")

print("\nTraining completed successfully!")
