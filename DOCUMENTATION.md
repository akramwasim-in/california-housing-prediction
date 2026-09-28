# Comprehensive Project Documentation
## California Housing Price Prediction & Web Deployment

---
## 1. Executive Summary
This project provides an end-to-end Machine Learning solution designed to predict median house prices across various California block groups based on geographic and structural parameters. The system encompasses dataset exploration, machine learning model evaluation, automated preprocessing pipelines, model serialization using Joblib, and real-time model serving via a Flask web application.

---
## 2. Problem Statement & Objectives
*Problem Type*: Supervised Learning - Regression
*Target Variable*: "median_house_value" (Continuous Numerical Feature)
*Primary Goal*: Build a robust regression model that minimizes Root Mean Squared Error (RMSE) to accurately predict housing prices on unseen data.
*Secondary Goal*: Serialize the trained pipeline and build a production-ready web interface for seamless end-user inference.

---
## 3. System Architecture & Directory Structure

### Project Layout
```text
project prediction/
│
├── static/
│   └── style.css            # Frontend styling sheet
├── templates/
│   ├── index.html           # Input form interface
│   └── predict.html         # Inference result display page
│
├── .gitignore               # Excludes large models (*.pkl) & temporary CSVs
├── Model_Evaluation.py      # Script for model comparison and cross-validation
├── persisting_model.py      # Final training, pipeline creation & serialization
├── app.py                   # Flask server entry point
├── housing.csv              # Raw dataset
├── README.md                # Quick repository overview
└── DOCUMENTATION.md         # Comprehensive technical report
```

## 4. Machine Learning Workflow
# 4.1 Data Ingestion & Stratified Splitting
Since median_income was identified as a critical continuous attribute for price estimation:

An income_category feature was engineered using pd.cut with bin edges ([0, 1.5, 3.0, 4.5, 6.0, np.inf]).

StratifiedShuffleSplit was utilized to divide the dataset into Training (80%) and Testing (20%) sets, preserving income category proportions across splits.

# 4.2 Data Preprocessing & Pipeline Construction
A modular Scikit-Learn Pipeline combined via ColumnTransformer was implemented:

Numerical Pipeline:

SimpleImputer(strategy="median"): Handles missing values (e.g., in total_bedrooms).

StandardScaler(): Standardizes feature scales to zero mean and unit variance.

Categorical Pipeline:

OneHotEncoder(handle_unknown="ignore"): Encodes nominal categories (ocean_proximity) into binary vectors.

# 4.3 Model Selection & Evaluation
Multiple regression algorithms were evaluated based on training RMSE and 10-Fold Cross-Validation:

Linear Regression: Served as the baseline benchmark model.

Decision Tree Regressor: Showed zero training error but suffered severe overfitting during cross-validation.

Random Forest Regressor: Delivered superior generalization with the lowest cross-validation RMSE and was selected as the final model.

## 5. Technical Implementation Details
# 5.1 Model Serialization (persisting_model.py)
Automated script verifies whether serialized model binaries (model.pkl) exist.

Executes full training and preprocessing workflows, persisting trained artifacts using joblib.dump().

Generates input.csv for downstream input schema validation.

# 5.2 Web Application (app.py)
Flask Framework: Loads pre-trained model.pkl and pipeline.pkl into memory upon startup.

GET / Route: Serves the user input form template (index.html).

POST /predict Route:

Extracts form attributes and constructs a Pandas DataFrame.

Transforms raw input features via pipeline.transform().

Performs inference and renders the predicted value on predict.html.

## 6. Setup & Execution Guide
1. Install Dependencies
Bash
pip install pandas numpy scikit-learn flask joblib
2. Generate Model Artifacts
Bash
python persisting_model.py
3. Launch Flask Application
Bash
python app.py
Access the application like at http://127.0.0.1:5000/.

