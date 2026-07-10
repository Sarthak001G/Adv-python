# Step 1: Import Libraries and Load Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("diabetes.csv")

print("First 5 rows:")
print(df.head())


# Step 2: Inspect Data Structure and Check Missing Values
print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# Step 3: Statistical Summary and Visualizing Outliers
print("\nStatistical Summary:")
print(df.describe())

fig, axs = plt.subplots(len(df.columns), 1, figsize=(7, 18), dpi=95)

for i, col in enumerate(df.columns):
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)

plt.tight_layout()
plt.show()


# Step 4: Remove Outliers Using IQR Method
q1, q3 = np.percentile(df["Insulin"], [25, 75])
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

clean_df = df[(df["Insulin"] >= lower) & (df["Insulin"] <= upper)]

print("\nData after removing outliers:")
print(clean_df.shape)


# Step 5: Correlation Analysis
corr = df.corr()

plt.figure(dpi=130)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

print("\nCorrelation with Outcome:")
print(corr["Outcome"].sort_values(ascending=False))


# Step 6: Visualize Target Variable Distribution
plt.pie(
    df["Outcome"].value_counts(),
    labels=["Diabetes", "Not Diabetes"],
    autopct="%.f%%",
    shadow=True
)

plt.title("Outcome Proportionality")
plt.show()


# Step 7: Separate Features and Target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# Step 8: Feature Scaling

# Normalization (Min-Max Scaling)
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X)

print("\nFirst 5 rows after MinMax Scaling:")
print(X_normalized[:5])


# Standardization
standard = StandardScaler()
X_standardized = standard.fit_transform(X)

print("\nFirst 5 rows after Standardization:")
print(X_standardized[:5])