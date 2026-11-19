# ============================================================
# LOAN APPROVAL PREDICTION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("E://Prudhwi_7//loan_approval_dataset.csv")

data.columns = data.columns.str.strip()

print("DATASET")
print(data.head())

print("\nShape:", data.shape)

print("\nColumn Names:")
print(data.columns.tolist())


# ============================================================
# 2. REMOVE ID COLUMN
# ============================================================

if "loan_id" in data.columns:
    data.drop("loan_id", axis=1, inplace=True)

if "Loan_ID" in data.columns:
    data.drop("Loan_ID", axis=1, inplace=True)


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

numeric_columns = data.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:
    data[column] = data[column].fillna(
        data[column].median()
    )


categorical_columns = data.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    data[column] = data[column].fillna(
        data[column].mode()[0]
    )


# ============================================================
# 4. CONVERT CATEGORICAL DATA
# ============================================================

# Encode input categorical columns separately
feature_encoders = {}

for column in data.select_dtypes(include=["object"]).columns:

    if column != "loan_status":

        le = LabelEncoder()

        data[column] = le.fit_transform(data[column])

        feature_encoders[column] = le


# Encode target column separately
status_encoder = LabelEncoder()

data["loan_status"] = status_encoder.fit_transform(
    data["loan_status"]
)


# ============================================================
# 5. INPUT AND OUTPUT
# ============================================================

X = data.drop("loan_status", axis=1)

y = data["loan_status"]


print("\nINPUT FEATURES:")
print(X.columns.tolist())

print("\nTARGET:")
print("loan_status")


# ============================================================
# 6. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)

print("Testing Data:", X_test.shape)


# ============================================================
# 7. STANDARDIZE DATA
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

lr = LogisticRegression()

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)


lr_acc = accuracy_score(y_test, lr_pred) * 100

lr_pre = precision_score(
    y_test, lr_pred,
    average="weighted",
    zero_division=0
) * 100

lr_rec = recall_score(
    y_test, lr_pred,
    average="weighted",
    zero_division=0
) * 100

lr_f1 = f1_score(
    y_test, lr_pred,
    average="weighted",
    zero_division=0
) * 100


print("\nLogistic Regression Accuracy:",
      round(lr_acc, 2), "%")


# ============================================================
# 9. DECISION TREE
# ============================================================

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)


dt_acc = accuracy_score(y_test, dt_pred) * 100

dt_pre = precision_score(
    y_test, dt_pred,
    average="weighted",
    zero_division=0
) * 100

dt_rec = recall_score(
    y_test, dt_pred,
    average="weighted",
    zero_division=0
) * 100

dt_f1 = f1_score(
    y_test, dt_pred,
    average="weighted",
    zero_division=0
) * 100


print("Decision Tree Accuracy:",
      round(dt_acc, 2), "%")


# ============================================================
# 10. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)


rf_acc = accuracy_score(y_test, rf_pred) * 100

rf_pre = precision_score(
    y_test, rf_pred,
    average="weighted",
    zero_division=0
) * 100

rf_rec = recall_score(
    y_test, rf_pred,
    average="weighted",
    zero_division=0
) * 100

rf_f1 = f1_score(
    y_test, rf_pred,
    average="weighted",
    zero_division=0
) * 100


print("Random Forest Accuracy:",
      round(rf_acc, 2), "%")


# ============================================================
# 11. K-NEAREST NEIGHBORS
# ============================================================

knn = KNeighborsClassifier()

knn.fit(X_train_scaled, y_train)

knn_pred = knn.predict(X_test_scaled)


knn_acc = accuracy_score(y_test, knn_pred) * 100

knn_pre = precision_score(
    y_test, knn_pred,
    average="weighted",
    zero_division=0
) * 100

knn_rec = recall_score(
    y_test, knn_pred,
    average="weighted",
    zero_division=0
) * 100

knn_f1 = f1_score(
    y_test, knn_pred,
    average="weighted",
    zero_division=0
) * 100


print("KNN Accuracy:",
      round(knn_acc, 2), "%")


# ============================================================
# 12. MODEL PERFORMANCE TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN"
    ],

    "Accuracy": [
        lr_acc, dt_acc, rf_acc, knn_acc
    ],

    "Precision": [
        lr_pre, dt_pre, rf_pre, knn_pre
    ],

    "Recall": [
        lr_rec, dt_rec, rf_rec, knn_rec
    ],

    "F1 Score": [
        lr_f1, dt_f1, rf_f1, knn_f1
    ]
})


print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(results.round(2).to_string(index=False))


# ============================================================
# 13. FIND BEST MODEL
# ============================================================

best_model = results.loc[
    results["Accuracy"].idxmax()
]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best Model:", best_model["Model"])

print("Accuracy:",
      round(best_model["Accuracy"], 2), "%")


# ============================================================
# 14. BAR GRAPH - MODEL ACCURACY
# ============================================================

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN"
]

accuracy = [
    lr_acc, dt_acc, rf_acc, knn_acc
]


plt.figure(figsize=(10, 6))

plt.bar(models, accuracy)

plt.title("Loan Approval Prediction - Model Accuracy")

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# ============================================================
# 15. ALL METRICS GRAPH
# ============================================================

results_plot = results.set_index("Model")

results_plot[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Loan Approval Prediction - Model Performance")

plt.xlabel("Models")

plt.ylabel("Score (%)")

plt.ylim(0, 100)

plt.xticks(rotation=15)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("LOAN APPROVAL PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# 17. NEW SAMPLE LOAN PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("NEW LOAN APPLICATION PREDICTION")
print("=" * 70)


# New sample applicant
new_sample = pd.DataFrame([{

    "no_of_dependents": 2,
    "education": 1,
    "self_employed": 0,
    "income_annum": 6000000,
    "loan_amount": 1500000,
    "loan_term": 12,
    "cibil_score": 750,
    "residential_assets_value": 2000000,
    "commercial_assets_value": 1000000,
    "luxury_assets_value": 3000000,
    "bank_asset_value": 1500000

}])


# Arrange columns in the same order as training data
new_sample = new_sample[X.columns]


# Scale sample
new_sample_scaled = scaler.transform(new_sample)


# Predict using all four models
lr_result = lr.predict(new_sample_scaled)[0]

dt_result = dt.predict(new_sample)[0]

rf_result = rf.predict(new_sample)[0]

knn_result = knn.predict(new_sample_scaled)[0]


# Convert encoded prediction into actual loan status
def get_status(result):

    return status_encoder.inverse_transform(
        [result]
    )[0]


# Display predictions with approval status
print("\nLogistic Regression:",
      get_status(lr_result))

print("Decision Tree:",
      get_status(dt_result))

print("Random Forest:",
      get_status(rf_result))

print("KNN:",
      get_status(knn_result))


print("\n")
print("=" * 70)
print("NEW LOAN PREDICTION COMPLETED")
print("=" * 70)