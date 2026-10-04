# ============================================================
# MCA201P - LAB SHEET 06
# MINI PROJECTS ON SUPERVISED LEARNING APPLICATIONS
# ============================================================

# ============================================================
# PROJECT 1: STUDENT PERFORMANCE PREDICTION SYSTEM
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 2. CREATE STUDENT DATASET
# ------------------------------------------------------------

student_data = {
    "Attendance": [85, 90, 75, 95, 80, 70, 88, 92, 78, 96,
                   84, 73, 89, 91, 76, 87, 94, 79, 82, 90],
    "Internal_Marks": [78, 85, 65, 92, 74, 60, 82, 88, 70, 95,
                       76, 62, 84, 86, 68, 80, 90, 72, 75, 87],
    "Assignments": [80, 88, 70, 95, 76, 65, 85, 90, 72, 96,
                    82, 68, 86, 89, 71, 84, 93, 75, 78, 91],
    "Study_Hours": [3, 5, 2, 6, 4, 1, 4, 5, 2, 7,
                    3, 2, 5, 6, 2, 4, 6, 3, 3, 5],
    "Previous_Semester": [75, 82, 68, 90, 72, 58, 80, 85, 69, 92,
                          78, 61, 83, 84, 66, 79, 88, 70, 73, 85],
    "Final_Marks": [78, 85, 67, 94, 75, 61, 83, 89, 71, 96,
                    80, 64, 85, 88, 69, 82, 91, 73, 76, 89]
}

student_df = pd.DataFrame(student_data)


# ------------------------------------------------------------
# 3. DISPLAY DATASET
# ------------------------------------------------------------

print(student_df.head())
print(student_df.info())
print(student_df.describe())


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print(student_df.isnull().sum())


# ------------------------------------------------------------
# 5. REMOVE DUPLICATE RECORDS
# ------------------------------------------------------------

student_df = student_df.drop_duplicates()


# ------------------------------------------------------------
# 6. VISUALIZE CORRELATION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
sns.heatmap(student_df.corr(), annot=True, cmap="coolwarm")
plt.title("Student Performance Correlation")
plt.show()


# ------------------------------------------------------------
# 7. PREPARE FEATURES AND TARGET
# ------------------------------------------------------------

X_student = student_df[
    [
        "Attendance",
        "Internal_Marks",
        "Assignments",
        "Study_Hours",
        "Previous_Semester"
    ]
]

y_student = student_df["Final_Marks"]


# ------------------------------------------------------------
# 8. SPLIT DATA INTO TRAINING AND TESTING
# ------------------------------------------------------------

X_train_student, X_test_student, y_train_student, y_test_student = (
    train_test_split(
        X_student,
        y_student,
        test_size=0.20,
        random_state=42
    )
)


# ------------------------------------------------------------
# 9. FEATURE SCALING
# ------------------------------------------------------------

student_scaler = StandardScaler()

X_train_student_scaled = student_scaler.fit_transform(
    X_train_student
)

X_test_student_scaled = student_scaler.transform(
    X_test_student
)


# ------------------------------------------------------------
# 10. LINEAR REGRESSION MODEL
# ------------------------------------------------------------

student_linear_model = LinearRegression()

student_linear_model.fit(
    X_train_student_scaled,
    y_train_student
)

student_linear_prediction = student_linear_model.predict(
    X_test_student_scaled
)


# ------------------------------------------------------------
# 11. EVALUATE LINEAR REGRESSION
# ------------------------------------------------------------

student_linear_mae = mean_absolute_error(
    y_test_student,
    student_linear_prediction
)

student_linear_mse = mean_squared_error(
    y_test_student,
    student_linear_prediction
)

student_linear_rmse = np.sqrt(student_linear_mse)

student_linear_r2 = r2_score(
    y_test_student,
    student_linear_prediction
)

print("Linear Regression MAE:",
      student_linear_mae)

print("Linear Regression MSE:",
      student_linear_mse)

print("Linear Regression RMSE:",
      student_linear_rmse)

print("Linear Regression R2:",
      student_linear_r2)


# ------------------------------------------------------------
# 12. RANDOM FOREST REGRESSION MODEL
# ------------------------------------------------------------

student_rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

student_rf_model.fit(
    X_train_student,
    y_train_student
)

student_rf_prediction = student_rf_model.predict(
    X_test_student
)


# ------------------------------------------------------------
# 13. EVALUATE RANDOM FOREST
# ------------------------------------------------------------

student_rf_mae = mean_absolute_error(
    y_test_student,
    student_rf_prediction
)

student_rf_mse = mean_squared_error(
    y_test_student,
    student_rf_prediction
)

student_rf_rmse = np.sqrt(student_rf_mse)

student_rf_r2 = r2_score(
    y_test_student,
    student_rf_prediction
)

print("Random Forest MAE:",
      student_rf_mae)

print("Random Forest MSE:",
      student_rf_mse)

print("Random Forest RMSE:",
      student_rf_rmse)

print("Random Forest R2:",
      student_rf_r2)


# ------------------------------------------------------------
# 14. COMPARE STUDENT MODELS
# ------------------------------------------------------------

student_model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        student_linear_mae,
        student_rf_mae
    ],
    "RMSE": [
        student_linear_rmse,
        student_rf_rmse
    ],
    "R2 Score": [
        student_linear_r2,
        student_rf_r2
    ]
})

print(student_model_comparison)


# ------------------------------------------------------------
# 15. VISUALIZE ACTUAL VS PREDICTED MARKS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_student,
    student_rf_prediction
)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Student Marks")

plt.plot(
    [y_test_student.min(), y_test_student.max()],
    [y_test_student.min(), y_test_student.max()],
    linestyle="--"
)

plt.show()


# ------------------------------------------------------------
# 16. PREDICT PERFORMANCE OF A NEW STUDENT
# ------------------------------------------------------------

new_student = pd.DataFrame({
    "Attendance": [88],
    "Internal_Marks": [82],
    "Assignments": [85],
    "Study_Hours": [4],
    "Previous_Semester": [80]
})

new_student_prediction = student_rf_model.predict(
    new_student
)

print("Predicted Final Marks:",
      new_student_prediction[0])


# ------------------------------------------------------------
# 17. SAVE STUDENT MODEL
# ------------------------------------------------------------

joblib.dump(
    student_rf_model,
    "student_performance_model.pkl"
)

joblib.dump(
    student_scaler,
    "student_scaler.pkl"
)


# ============================================================
# PROJECT 2: HOUSE PRICE PREDICTION SYSTEM
# ============================================================

# ------------------------------------------------------------
# 18. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 19. CREATE HOUSE DATASET
# ------------------------------------------------------------

house_data = {
    "Area": [
        1200, 1500, 1800, 1000, 2200,
        1600, 2000, 1300, 1700, 2500,
        1100, 1900, 1400, 2100, 1550,
        2300, 1250, 1750, 1950, 1450
    ],

    "Bedrooms": [
        2, 3, 3, 2, 4,
        3, 4, 2, 3, 5,
        2, 4, 3, 4, 3,
        4, 2, 3, 4, 3
    ],

    "Location": [
        "Roorkee", "Haridwar", "Dehradun", "Roorkee",
        "Dehradun", "Haridwar", "Dehradun", "Roorkee",
        "Haridwar", "Dehradun", "Roorkee", "Haridwar",
        "Dehradun", "Roorkee", "Haridwar", "Dehradun",
        "Roorkee", "Haridwar", "Dehradun", "Roorkee"
    ],

    "Bathrooms": [
        1, 2, 2, 1, 3,
        2, 3, 1, 2, 3,
        1, 3, 2, 3, 2,
        3, 1, 2, 3, 2
    ],

    "Parking": [
        1, 1, 2, 0, 2,
        1, 2, 1, 1, 2,
        0, 2, 1, 2, 1,
        2, 1, 1, 2, 1
    ],

    "Age": [
        10, 5, 3, 15, 2,
        7, 4, 12, 6, 1,
        18, 5, 8, 3, 9,
        2, 14, 6, 4, 10
    ],

    "Price": [
        42, 58, 72, 34, 95,
        65, 88, 45, 62, 110,
        30, 80, 55, 92, 60,
        100, 40, 68, 85, 52
    ]
}

house_df = pd.DataFrame(house_data)


# ------------------------------------------------------------
# 20. DISPLAY HOUSE DATASET
# ------------------------------------------------------------

print(house_df.head())
print(house_df.info())
print(house_df.describe())


# ------------------------------------------------------------
# 21. CHECK MISSING VALUES
# ------------------------------------------------------------

print(house_df.isnull().sum())


# ------------------------------------------------------------
# 22. REMOVE DUPLICATE RECORDS
# ------------------------------------------------------------

house_df = house_df.drop_duplicates()


# ------------------------------------------------------------
# 23. VISUALIZE HOUSE PRICE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    house_df["Price"],
    bins=8
)

plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.title("House Price Distribution")

plt.show()


# ------------------------------------------------------------
# 24. AREA VS HOUSE PRICE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    house_df["Area"],
    house_df["Price"]
)

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs House Price")

plt.show()


# ------------------------------------------------------------
# 25. CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_house_df = house_df[
    [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Parking",
        "Age",
        "Price"
    ]
]

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_house_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("House Price Correlation")

plt.show()


# ------------------------------------------------------------
# 26. PREPARE FEATURES AND TARGET
# ------------------------------------------------------------

X_house = house_df[
    [
        "Area",
        "Bedrooms",
        "Location",
        "Bathrooms",
        "Parking",
        "Age"
    ]
]

y_house = house_df["Price"]


# ------------------------------------------------------------
# 27. DEFINE CATEGORICAL AND NUMERICAL FEATURES
# ------------------------------------------------------------

numerical_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Parking",
    "Age"
]

categorical_features = [
    "Location"
]


# ------------------------------------------------------------
# 28. PREPROCESSING PIPELINE
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 29. SPLIT HOUSE DATA
# ------------------------------------------------------------

X_train_house, X_test_house, y_train_house, y_test_house = (
    train_test_split(
        X_house,
        y_house,
        test_size=0.20,
        random_state=42
    )
)


# ------------------------------------------------------------
# 30. LINEAR REGRESSION PIPELINE
# ------------------------------------------------------------

house_linear_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

house_linear_pipeline.fit(
    X_train_house,
    y_train_house
)

house_linear_prediction = house_linear_pipeline.predict(
    X_test_house
)


# ------------------------------------------------------------
# 31. EVALUATE LINEAR REGRESSION
# ------------------------------------------------------------

house_linear_mae = mean_absolute_error(
    y_test_house,
    house_linear_prediction
)

house_linear_mse = mean_squared_error(
    y_test_house,
    house_linear_prediction
)

house_linear_rmse = np.sqrt(
    house_linear_mse
)

house_linear_r2 = r2_score(
    y_test_house,
    house_linear_prediction
)

print("House Linear Regression MAE:",
      house_linear_mae)

print("House Linear Regression MSE:",
      house_linear_mse)

print("House Linear Regression RMSE:",
      house_linear_rmse)

print("House Linear Regression R2:",
      house_linear_r2)


# ------------------------------------------------------------
# 32. RANDOM FOREST REGRESSION PIPELINE
# ------------------------------------------------------------

house_rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

house_rf_pipeline.fit(
    X_train_house,
    y_train_house
)

house_rf_prediction = house_rf_pipeline.predict(
    X_test_house
)


# ------------------------------------------------------------
# 33. EVALUATE RANDOM FOREST
# ------------------------------------------------------------

house_rf_mae = mean_absolute_error(
    y_test_house,
    house_rf_prediction
)

house_rf_mse = mean_squared_error(
    y_test_house,
    house_rf_prediction
)

house_rf_rmse = np.sqrt(
    house_rf_mse
)

house_rf_r2 = r2_score(
    y_test_house,
    house_rf_prediction
)

print("House Random Forest MAE:",
      house_rf_mae)

print("House Random Forest MSE:",
      house_rf_mse)

print("House Random Forest RMSE:",
      house_rf_rmse)

print("House Random Forest R2:",
      house_rf_r2)


# ------------------------------------------------------------
# 34. COMPARE HOUSE PRICE MODELS
# ------------------------------------------------------------

house_model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        house_linear_mae,
        house_rf_mae
    ],
    "RMSE": [
        house_linear_rmse,
        house_rf_rmse
    ],
    "R2 Score": [
        house_linear_r2,
        house_rf_r2
    ]
})

print(house_model_comparison)


# ------------------------------------------------------------
# 35. PREDICT PRICE OF A NEW HOUSE
# ------------------------------------------------------------

new_house = pd.DataFrame({
    "Area": [1800],
    "Bedrooms": [3],
    "Location": ["Dehradun"],
    "Bathrooms": [2],
    "Parking": [1],
    "Age": [5]
})

predicted_house_price = house_rf_pipeline.predict(
    new_house
)

print(
    "Predicted House Price:",
    predicted_house_price[0]
)


# ------------------------------------------------------------
# 36. ACTUAL VS PREDICTED HOUSE PRICES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_house,
    house_rf_prediction
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test_house.min(), y_test_house.max()],
    [y_test_house.min(), y_test_house.max()],
    linestyle="--"
)

plt.show()


# ------------------------------------------------------------
# 37. SAVE HOUSE PRICE MODEL
# ------------------------------------------------------------

joblib.dump(
    house_rf_pipeline,
    "house_price_prediction_model.pkl"
)


# ------------------------------------------------------------
# 38. LOAD SAVED HOUSE PRICE MODEL
# ------------------------------------------------------------

loaded_house_model = joblib.load(
    "house_price_prediction_model.pkl"
)


# ------------------------------------------------------------
# 39. PREDICT USING SAVED MODEL
# ------------------------------------------------------------

saved_model_prediction = loaded_house_model.predict(
    new_house
)

print(
    "Prediction using saved model:",
    saved_model_prediction[0]
)


# ------------------------------------------------------------
# 40. FINAL MODEL COMPARISON
# ------------------------------------------------------------

print("======================================")
print("STUDENT PERFORMANCE PROJECT")
print("======================================")

print(student_model_comparison)

print("======================================")
print("HOUSE PRICE PREDICTION PROJECT")
print("======================================")

print(house_model_comparison)
