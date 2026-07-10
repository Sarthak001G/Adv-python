from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#1 Load the Dataset
housing = pd.read_csv("housing.csv")

#2 Create a Stratified Shuffle test set
housing['income_cat'] = pd.cut(housing['median_income'],
                             bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                             labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

#We will work on the  copy of training data
housing = strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# print(housing, housing_labels)


# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]



# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
 
# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
 
# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
 
# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
 
# housing_prepared is now a NumPy array ready for training
print(housing_prepared.shape)
# 7. Train models
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
lin_rmse =-np.sqrt(-cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)).mean()
print(f"Linear Regression RMSE: {lin_rmse}")

# Decision Tree Regressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
tree_pred = tree_reg.predict(housing_prepared)
# tree_rmse = np.sqrt(mean_squared_error(housing_labels, tree_pred))
dec_rmses=-np.sqrt(-cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10))
tree_rmse = dec_rmses.mean()
print(pd.Series(dec_rmses).describe())

# Random forest Regressor
rf_reg = RandomForestRegressor( random_state=42)
rf_reg.fit(housing_prepared, housing_labels)
rf_pred = rf_reg.predict(housing_prepared)
rf_rmse =-np.sqrt(-cross_val_score(rf_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)).mean()
print(f"Random forest RMSE: {rf_rmse}")