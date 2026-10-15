import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. Load the dataset
# ==========================================

print("Loading dataset...")

data = pd.read_csv("tips.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(data))
print("Columns:", data.columns.tolist())


# ==========================================
# 2. Separate input and target
# ==========================================

X = data.drop("tip", axis=1)

y = data["tip"]


# ==========================================
# 3. Define columns
# ==========================================

categorical_columns = [
    "sex",
    "smoker",
    "day",
    "time"
]

numeric_columns = [
    "total_bill",
    "size"
]


# ==========================================
# 4. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numeric",
            "passthrough",
            numeric_columns
        )
    ]
)


# ==========================================
# 5. Create machine learning model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 6. Create pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 7. Split the data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 8. Train the model
# ==========================================

print("Training model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 9. Test the model
# ==========================================

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)


print()
print("================================")
print("MODEL RESULTS")
print("================================")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# ==========================================
# 10. Save the trained model
# ==========================================

joblib.dump(pipeline, "tip_model.pkl")

print()
print("================================")
print("SUCCESS!")
print("Model saved as tip_model.pkl")
print("================================")