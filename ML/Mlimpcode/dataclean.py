# Step 1: Import Libraries and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Titanic-Dataset.csv")

print(df.info())
print(df.head())

# Step 2: Check for Duplicate Rows
print("Duplicate rows:")
print(df.duplicated())

# Step 3: Identify Column Data Types
cat_col = [col for col in df.columns if df[col].dtype == "object"]
num_col = [col for col in df.columns if df[col].dtype != "object"]

print("Categorical columns:", cat_col)
print("Numerical columns:", num_col)

# Step 4: Count Unique Values in Categorical Columns
print("Unique values in categorical columns:")
print(df[cat_col].nunique())

# Step 5: Calculate Missing Values Percentage
missing_percentage = round((df.isnull().sum() / df.shape[0]) * 100, 2)
print("Missing value percentage:")
print(missing_percentage)

# Step 6: Drop Irrelevant Columns and Handle Missing Data
df1 = df.drop(columns=["Name", "Ticket", "Cabin"])
df1.dropna(subset=["Embarked"], inplace=True)

df1["Age"].fillna(df1["Age"].mean(), inplace=True)

# Step 7: Detect Outliers using Box Plot
plt.boxplot(df1["Age"], vert=False)
plt.ylabel("Variable")
plt.xlabel("Age")
plt.title("Box Plot")
plt.show()

# Step 8: Calculate Outlier Boundaries and Remove Them
mean = df1["Age"].mean()
std = df1["Age"].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

df2 = df1[(df1["Age"] >= lower_bound) & (df1["Age"] <= upper_bound)]

# Step 9: Impute Missing Data Again
df3 = df2.fillna(df2["Age"].mean())
print("Missing values after cleaning:")
print(df3.isnull().sum())

# Step 10: Recalculate Outlier Bounds
mean = df3["Age"].mean()
std = df3["Age"].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

df4 = df3[(df3["Age"] >= lower_bound) & (df3["Age"] <= upper_bound)]

# Step 11: Data Validation and Feature Selection
X = df3[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
Y = df3["Survived"]

print("Features:")
print(X.head())

print("Target:")
print(Y.head())

# Step 12: Data Formatting (Min-Max Scaling)

scaler = MinMaxScaler(feature_range=(0, 1))

num_col_ = [col for col in X.columns if X[col].dtype != "object"]

x1 = X.copy()
x1[num_col_] = scaler.fit_transform(x1[num_col_])

print("Scaled Data:")
print(x1.head())