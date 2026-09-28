# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


# Importing dataset
housing = pd.read_csv("housing.csv")
housing["income_category"] = pd.cut(housing["median_income"],
                                    bins=[0,1.5,3.0,4.5,6.0,np.inf],
                                    labels=["a","b","c","d","e"])
# Splitting the data
split = StratifiedShuffleSplit()
for train_index , test_index in split.split(housing,housing["income_category"]):
    housing_train_data = housing.iloc[train_index]
    housing_test_data = housing.iloc[test_index]

housing = housing.drop(columns=["income_category"])


# Seperate Feature and labels
housing_labels = housing_train_data["median_house_value"]
housing_features = housing_train_data.drop(columns=["median_house_value","income_category"]).copy()

# Creating a pipeline
## preparing data seperately
housing_features_num = housing_features.select_dtypes(include = [np.number]).columns.to_list()
housing_features_cat = ["ocean_proximity"]


### pipeline for number attribute
num_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
### pipeline for categorical attribute
cat_pipeline = Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

### create a full pipeline
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,housing_features_num),
    ("one-hot",cat_pipeline,housing_features_cat)
])

# Transform the data
prepared_data = full_pipeline.fit_transform(housing_features)

# Train the model
## Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(prepared_data,housing_labels)
lin_pred = lin_reg.predict(prepared_data)
lin_rmse = root_mean_squared_error(housing_labels,lin_pred)
lin_cross = -cross_val_score(lin_reg,prepared_data,housing_labels,
                            scoring="neg_root_mean_squared_error",cv=10)

## Decision tree regressor
des_tree = DecisionTreeRegressor()
des_tree.fit(prepared_data,housing_labels)
des_tree_pred = des_tree.predict(prepared_data)
des_tree_rmse = root_mean_squared_error(housing_labels,des_tree_pred)
des_cross = -cross_val_score(des_tree,prepared_data,housing_labels,
                            scoring="neg_root_mean_squared_error",cv=10)
## Random Forest Refressor
rf = RandomForestRegressor()
rf.fit(prepared_data,housing_labels)
rf_pred = rf.predict(prepared_data)
rf_rmse = root_mean_squared_error(housing_labels,rf_pred)
rf_cross = -cross_val_score(rf,prepared_data,housing_labels,
                            scoring="neg_root_mean_squared_error",cv=10)

# - ------------------------
print("Linear Regression Performance : ")
print(lin_rmse)

print("-------------------------------------")
print("Decision Tree Regressor Performance : ")
print(des_tree_rmse)

print("-------------------------------------")
print("Random Forest Regressor Performance : ")
print(rf_rmse)


print("-------------------------------------------")
print("Cross Validation score for Linear Regression is")
print(lin_cross)

print("-------------------------------------------")
print("Cross Validation score for Decision Regression is")
print(des_cross)

print("-------------------------------------------")
print("Cross Validation score for Random Regression is")
print(rf_cross)