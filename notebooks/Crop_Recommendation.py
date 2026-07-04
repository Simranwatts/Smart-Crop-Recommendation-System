# ==========================================
# SMART CROP RECOMMENDATION SYSTEM
# ==========================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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