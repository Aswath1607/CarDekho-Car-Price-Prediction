from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "car_name": [request.form["car_name"]],
        "brand": [request.form["brand"]],
        "model": [request.form["model"]],
        "vehicle_age": [int(request.form["vehicle_age"])],
        "km_driven": [int(request.form["km_driven"])],
        "seller_type": [request.form["seller_type"]],
        "fuel_type": [request.form["fuel_type"]],
        "transmission_type": [request.form["transmission_type"]],
        "mileage": [float(request.form["mileage"])],
        "engine": [float(request.form["engine"])],
        "max_power": [float(request.form["max_power"])],
        "seats": [int(request.form["seats"])]
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated Selling Price: ₹ {prediction:,.0f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
    