# Predictive Modeling Using Machine Learning
# Heart Disease Prediction using Random Forest

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 1. Load dataset
df = pd.read_csv("dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 2. Handle missing values
numeric_columns = df.select_dtypes(include="number").columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 3. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Create and train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

print("\nModel training completed.")

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Heart Disease Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 9. Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 6))
importance.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

print("\nProject completed successfully!")
