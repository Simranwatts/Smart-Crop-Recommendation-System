# ==========================================
# SMART CROP RECOMMENDATION SYSTEM
# ==========================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("dataset/Crop_Recommendation.csv")

# Display First 5 Rows
print("First 5 Rows of Dataset")
print(df.head())

# Display Dataset Shape
print("\nShape of Dataset:")
print(df.shape)

# Display Column Names
print("\nColumns:")
print(df.columns)

# Display Dataset Information
print("\nDataset Information:")
df.info()

# Display Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Crop Distribution
# ==========================================
plt.figure(figsize=(12,6))

sns.countplot(data=df, x="Crop")

plt.title("Distribution of Crops")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=90)

plt.savefig("images/crop_distribution.png")
plt.show()

# ==========================================
# Correlation Heatmap
# ==========================================
plt.figure(figsize=(10,8))

correlation = df.drop("Crop", axis=1).corr()

sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f")

plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# ==========================================
# Feature Selection
# ==========================================
# Input Features
X = df.drop("Crop", axis=1)

# Target Variable
y = df["Crop"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFirst 5 Features:")
print(X.head())

print("\nFirst 5 Target Values:")
print(y.head())

# ==========================================
# Label Encoding
# ==========================================
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nEncoded Target Values:")
print(y[:10])

# ==========================================
# Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# ==========================================
# Decision Tree Classifier
# ==========================================
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the model
decision_tree.fit(X_train, y_train)

# Make predictions
y_pred_dt = decision_tree.predict(X_test)

# Calculate Accuracy
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\n========== Decision Tree ==========")
print(f"Accuracy : {dt_accuracy:.4f}")

# ==========================================
# K-Nearest Neighbors (KNN)
# ==========================================
# Create Model
knn = KNeighborsClassifier(n_neighbors=5)

# Train Model
knn.fit(X_train, y_train)

# Predict on Test Data
y_pred_knn = knn.predict(X_test)

# Calculate Accuracy
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\n========== K-Nearest Neighbors ==========")
print(f"Accuracy : {knn_accuracy:.4f}")

# ==========================================
# Random Forest Classifier
# ==========================================
# Create Model
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
random_forest.fit(X_train, y_train)

# Predict on Test Data
y_pred_rf = random_forest.predict(X_test)

# Calculate Accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\n========== Random Forest ==========")
print(f"Accuracy : {rf_accuracy:.4f}")

# ==========================================
# Model Comparison
# ==========================================
print("\n======================================")
print("        Model Comparison")
print("======================================")

print(f"Decision Tree          : {dt_accuracy:.4f}")
print(f"K-Nearest Neighbors    : {knn_accuracy:.4f}")
print(f"Random Forest          : {rf_accuracy:.4f}")

# Select Best Model
if rf_accuracy > dt_accuracy and rf_accuracy > knn_accuracy:
    best_model = "Random Forest"
elif dt_accuracy > knn_accuracy:
    best_model = "Decision Tree"
else:
    best_model = "K-Nearest Neighbors"

print("\nBest Model:", best_model)