# Documentation
## California Housing Price Prediction & Web Deployment

---

## 1. Executive Summary
This project provides an end-to-end Machine Learning solution designed to predict median house prices across California block groups based on geographic, structural, and demographic parameters. The system encompasses dataset exploration, machine learning model evaluation, automated preprocessing pipelines, model serialization using Joblib, and real-time model serving via a Flask web application.

---

## 2. Problem Statement & Objectives
* **Problem Type**: Supervised Learning - Regression
* **Target Variable**: `median_house_value` (Continuous Numerical Feature)
* **Primary Goal**: Build a robust regression model that minimizes Root Mean Squared Error (RMSE) to accurately predict housing prices on unseen data.
* **Secondary Goal**: Serialize the trained pipeline and build a production-ready web interface for seamless end-user inference.

---

## 3. Dataset Schema & Overview

The underlying dataset `housing.csv` contains census information from California block groups.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `longitude` | Continuous (Float) | How far west the location is. |
| `latitude` | Continuous (Float) | How far north the location is. |
| `housing_median_age` | Continuous (Float) | Median age of houses in the block group. |
| `total_rooms` | Continuous (Float) | Total number of rooms in the block group. |
| `total_bedrooms` | Continuous (Float) | Total number of bedrooms (contains missing values). |
| `population` | Continuous (Float) | Total number of residents in the block group. |
| `households` | Continuous (Float) | Total number of households in the block group. |
| `median_income` | Continuous (Float) | Median household income (measured in tens of thousands of USD). |
| `ocean_proximity` | Categorical (String) | Geographic location relative to ocean (`NEAR BAY`, `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `ISLAND`). |
| `median_house_value` | Continuous (Float) | **Target Variable**: Median price of the house (in USD). |

---

## 4. System Architecture & Directory Structure

```text
project prediction/
│
├── static/
│   └── style.css            # Frontend styling sheet
├── templates/
│   ├── index.html           # Landing page interface
│   └── predict.html         # User input form & prediction display page
│
├── .gitignore               # Excludes large binaries (*.pkl) & temporary CSVs
├── Model_Evaluation.py      # Script for model comparison and 10-fold cross-validation
├── persisting_model.py      # Script for training, pipeline creation & serialization
├── app.py                   # Main Flask backend application server
├── housing.csv              # Raw California housing dataset
├── README.md                # Quick repository overview
└── DOCUMENTATION.md         # report
```
## 5. Machine Learning Workflow
### 5.1 Data Ingestion & Stratified Splitting
Since median_income was identified as a critical attribute for price estimation, an income_category feature was engineered using pd.cut() with bin edges [0, 1.5, 3.0, 4.5, 6.0, np.inf].

StratifiedShuffleSplit was utilized to divide the dataset into Training (80%) and Testing (20%) sets, preserving income category proportions across splits to prevent sampling bias.

### 5.2 Data Preprocessing & Pipeline Construction
A modular Scikit-Learn Pipeline combined via ColumnTransformer was implemented to avoid data leakage:

Numerical Pipeline:

SimpleImputer(strategy="median"): Handles missing values in total_bedrooms.

StandardScaler(): Standardizes feature scales to zero mean and unit variance.

Categorical Pipeline:

OneHotEncoder(handle_unknown="ignore"): Encodes nominal categories (ocean_proximity) into binary vectors.

### 5.3 Model Selection & Evaluation
Multiple regression algorithms were evaluated based on training RMSE and 10-Fold Cross-Validation:

Linear Regression: Used as a simple base model for comparison.

Decision Tree Regressor: Showed zero training error but suffered severe overfitting during cross-validation.

Random Forest Regressor: Delivered superior generalization with the lowest cross-validation RMSE and was selected as the production model.

## 6. Technical Implementation Details
### 6.1 Model Serialization (persisting_model.py)
The automated script verifies whether serialized model binaries (model.pkl) exist.

Executes full training and preprocessing workflows, persisting trained artifacts using joblib.dump().

Generates input.csv for downstream testing and input schema validation.

### 6.2 Web Application (app.py)
Flask Framework: Loads pre-trained model.pkl and pipeline.pkl into memory upon startup.

GET / Route: Serves the user landing page template (index.html).

POST /predict Route:

Extracts form attributes and constructs a Pandas DataFrame.

Transforms raw input features via pipeline.transform().

Performs inference using model.predict() and renders the estimated price on predict.html.

## 7. Setup & Execution Guide
Step 1: Install Dependencies
Bash
pip install pandas numpy scikit-learn flask joblib
Step 2: Generate Model Artifacts
Run the script to train and export model.pkl and pipeline.pkl:

Bash
python persisting_model.py
Step 3: Launch Flask Application
Bash
python app.py
Access the application in your browser at http://127.0.0.1:5000/.

## 8. Conclusion & Future Scope
Summary: The project successfully establishes an automated, end-to-end Machine Learning pipeline that takes user property parameters via a web interface and delivers reliable real-time house value estimations using a trained Random Forest model.

Future Scope:

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV for further RMSE reduction.
