# Importing Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

model_file='model.pkl'
pipeline_file = 'pipeline.pkl'

def build_pipeline(housing_features_num,housing_features_cat):
    ### pipeline for number attribute
    num_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    ### pipeline for categorical attribute
    cat_pipeline = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
    ("num",num_pipeline,housing_features_num),
    ("one-hot",cat_pipeline,housing_features_cat)
    ])
    return full_pipeline

if not(os.path.exists(model_file)):
    # Importing dataset
    housing = pd.read_csv("housing.csv")
    housing["income_category"] = pd.cut(housing["median_income"],
                                        bins=[0,1.5,3.0,4.5,6.0,np.inf],
                                        labels=["a","b","c","d","e"])
    # Splitting the data
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.20,random_state=42)
    for train_index , test_index in split.split(housing,housing["income_category"]):
        housing_train_data = housing.iloc[train_index]
        housing_test_data = housing.iloc[test_index]

    # save testing data to seperate file
    housing_test_data = housing_test_data.drop(columns=["income_category"])
    housing_test_data.to_csv('input.csv',index=False)

    housing = housing.drop(columns=["income_category"])

    # Seperate Feature and labels
    housing_labels = housing_train_data["median_house_value"]
    housing_features = housing_train_data.drop(columns=["median_house_value","income_category"]).copy()

    cat_attribute=['ocean_proximity']
    num_attribute=housing_features.select_dtypes(include = [np.number]).columns.to_list()

    # Pipeline creation
    pipeline = build_pipeline(num_attribute,cat_attribute)
    prepared_data = pipeline.fit_transform(housing_features)

    # Model calling
    model= RandomForestRegressor()
    model.fit(prepared_data,housing_labels) # trains the model

    joblib.dump(model,model_file)
    joblib.dump(pipeline,pipeline_file)
    print("Model Trained Succesfully")

else:
    model = joblib.load(model_file) # 1
    pipeline = joblib.load(pipeline_file) # 2
    input_data_orig=pd.read_csv('input.csv')
    input_data = input_data_orig.drop(columns=['median_house_value'])
    prepared_input_data = pipeline.transform(input_data) # 4 
    # print(input_data['ocean_proximity'].value_counts())
    predictions = model.predict(prepared_input_data) # 5
    input_data_orig['predicted value'] = predictions
    input_data_orig.to_csv('ouput.csv',index=False)
    print("Predictions / Inference completed")