from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and preprocessing pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None

    if request.method == "POST":

        longitude = float(request.form["longitude"])
        latitude = float(request.form["latitude"])
        housing_median_age = float(request.form["housing_median_age"])
        total_rooms = float(request.form["total_rooms"])
        total_bedrooms = float(request.form["total_bedrooms"])
        population = float(request.form["population"])
        households = float(request.form["households"])
        median_income = float(request.form["median_income"])
        ocean_proximity = request.form["ocean_proximity"]

        # Create dataframe with SAME column names as training data
        input_data = pd.DataFrame({
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity": [ocean_proximity]
        })

        # Use the SAME pipeline used during training
        prepared_data = pipeline.transform(input_data)

        # Predict
        result = model.predict(prepared_data)

        prediction = round(result[0], 2)

    return render_template(
        "predict.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)