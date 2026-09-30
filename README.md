# California Housing Price Prediction & Web Deployment

An end-to-end Machine Learning web application designed to predict median housing prices in California block groups. Built using Python, Scikit-Learn pipelines, Random Forest Regressor, and deployed locally via a Flask web application with custom HTML5/CSS3 frontend interfaces.

---

## 1. Project Overview & Problem Statement

### Project Overview
This project delivers a complete Machine Learning lifecycle solution—ranging from raw data ingestion, preprocessing, cross-validation model selection, artifact serialization, to real-time model serving.

### Problem Statement
Housing price estimation is a classic Supervised Learning Regression problem. Real estate market prices are influenced by multiple geographic, structural, and demographic factors. The objective of this project is to build an automated machine learning regression model that accurately estimates the continuous target variable (`median_house_value`) and provides predictions via an interactive web interface.

---

## 2. Dataset Overview & Schema

The underlying dataset `housing.csv` contains census information from California block groups.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `longitude` | Continuous (Float) | measuring distance west. |
| `latitude` | Continuous (Float) | measuring distance north. |
| `housing_median_age` | Continuous (Float) | Median age of houses group. |
| `total_rooms` | Continuous (Float) | Total rooms . |
| `total_bedrooms` | Continuous (Float) | Total bedrooms (contains missing values). |
| `population` | Continuous (Float) | Total resident population in a block group. |
| `households` | Continuous (Float) | Total households . |
| `median_income` | Continuous (Float) | Median household income (measured in tens of thousands of USD). |
| `ocean_proximity` | Categorical (String) | Geographic location relative to ocean (`NEAR BAY`, `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `ISLAND`). |
| `median_house_value` | Continuous (Float) | **Target Variable**: Median house value for households within a block group (USD). |

---

## 3. Machine Learning Workflow & Pipeline

### Step 1: Stratified Data Splitting
To ensure the training and testing datasets represent all income categories, `median_income` was binned into 5 categories using `pd.cut()`. We performed **Stratified Shuffle Split** (`StratifiedShuffleSplit`) with an 80:20 train-test ratio, ensuring balanced distribution across both sets.

### Step 2: Data Preprocessing & Custom Pipeline
We constructed a modular preprocessing pipeline using `ColumnTransformer` and Scikit-Learn `Pipeline`:
* **Numerical Pipeline (`num_pipeline`)**:
  * `SimpleImputer(strategy="median")`: Handles missing values in `total_bedrooms`.
  * `StandardScaler()`: Normalizes features to standard normal distribution (zero mean and unit variance).
* **Categorical Pipeline (`cat_pipeline`)**:
  * `OneHotEncoder(handle_unknown="ignore")`: Converts the string column `ocean_proximity` into binary vectors.

### Step 3: Model Evaluation & Selection
Three distinct algorithms were evaluated using Root Mean Squared Error (RMSE) and 10-Fold Cross-Validation:
1. **Linear Regression**: Baseline benchmark estimator.
2. **Decision Tree Regressor**: Exhibited severe overfitting on training data (high cross-validation error).
3. **Random Forest Regressor**: Achieved superior cross-validation performance and generalization stability; selected for production deployment.

### Step 4: Model Persistence
The final trained `RandomForestRegressor` and fitted preprocessing pipeline were serialized using `joblib`:
* `model.pkl`: Serialized model parameters.
* `pipeline.pkl`: Serialized data transformation pipeline.

---

## 4. Local Web Deployment

The application utilizes a Flask web framework for backend serving and custom HTML/CSS templates for client interactions.
* `app.py`: Loads `model.pkl` and `pipeline.pkl` into memory upon application launch.
* `predict.html`: Accepts user inputs via structured form fields.
* Inference Engine: Constructs a Pandas DataFrame from user inputs, transforms data via `pipeline.transform()`, and computes real-time predictions via `model.predict()`.

---

## 5. Technology Stack

* Programming Language: Python 3.8+
* Data Analysis & Machine Learning: Pandas, NumPy, Scikit-Learn
* Model Serialization: Joblib
* Web Backend Framework: Flask
* Frontend Technologies: HTML5, CSS3

---

## 6. Repository Structure

```text
project prediction/
│
├── static/
│   └── style.css            # Custom CSS stylesheet
├── templates/
│   ├── index.html           # Landing page template
│   └── predict.html         # Form input and prediction rendering template
├── .gitignore               # Excludes binaries (*.pkl) and generated outputs
├── Model_Evaluation.py      # Model comparison and cross-validation script
├── persisting_model.py      # Final pipeline build and model serialization script
├── app.py                   # Main Flask backend application server
├── housing.csv              # California housing dataset
├── README.md                # Project documentation
└── DOCUMENTATION.md         # About project
```
## 7. Setup & Local Execution Guide
Prerequisites
Python 3.8 or higher installed on your system.

Step 1: Clone Repository
Bash
git clone https://github.com/akramwasim-in/california-housing-prediction.git
cd california-housing-prediction
Step 2: Install Dependencies
Bash
pip install pandas numpy scikit-learn flask joblib
Step 3: Train and Export Model Artifacts
Execute the persistence script to generate model.pkl and pipeline.pkl:

Bash
python persisting_model.py
Step 4: Launch Web Application
Bash
python app.py
Step 5: Access Web Interface
Open a web browser and navigate to like (http://127.0.0.1:5000/).
