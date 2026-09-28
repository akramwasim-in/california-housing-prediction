## California Housing Price Prediction 

An end-to-end Machine Learning web application that predicts median housing prices in California using custom Scikit-Learn pipelines, Random Forest Regressor, and a Flask interface.

---

## Project Overview
This project addresses a regression problem: estimating house values based on various demographic, location, and structural features. It includes full dataset preprocessing, model evaluation using 10-fold cross-validation, persistence with Joblib, and model serving via a Flask web application.

---

## Repository Structure
```text
project prediction/
│
├── static/
│   └── style.css            # Styling for web templates
├── templates/
│   ├── index.html           # Landing page
│   └── predict.html         # Prediction result page
│
├── .gitignore               # Excludes large binaries & output files
├── Model_Evaluation.py      # Linear, Decision Tree, and Random Forest evaluation
├── persisting_model.py      # Script to build pipeline and export trained models
├── app.py                   # Flask server handling web requests and inference
└── housing.csv              # California housing dataset

## Machine Learning Pipeline & Workflow
Stratified Train-Test Split: Data is split based on income categories (median_income) to maintain class proportions.

Preprocessing Pipeline:

Numerical Features (StandardScaler, SimpleImputer using median strategy).

Categorical Features (OneHotEncoder for ocean_proximity).

Combined using ColumnTransformer.

Model Selection: Evaluated models using Root Mean Squared Error (RMSE) and 10-Fold Cross Validation. RandomForestRegressor delivered the best performance and was selected for production serialization.

## How to Run Locally
1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

2. Clone the Repository
Bash
git clone [https://github.com/akramwasim-in/california-housing-prediction.git]
cd "project prediction"
3. Install Required Dependencies
Bash
pip install pandas numpy scikit-learn flask joblib
4. Train and Persist the Model
Run the model persistence script to generate model.pkl and pipeline.pkl:

Bash
python persisting_model.py
5. Start the Flask App
Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000/.

## Tech Stack
Language: Python

Data Science / ML: Pandas, NumPy, Scikit-Learn

Web Backend: Flask

Frontend: HTML5, CSS3

Model Serialization: Joblib