# 🚗 Car Price Prediction Using Machine Learning

## Project Overview

This project predicts the selling price of a used car using Machine Learning. A Random Forest Regressor model was trained on the CarDekho dataset and deployed using Flask.

## Dataset

- Source: CarDekho Used Car Dataset
- Total Records: 15,411
- Features: 13 input features

## Features Used

- Car Name
- Brand
- Model
- Vehicle Age
- Kilometers Driven
- Seller Type
- Fuel Type
- Transmission Type
- Mileage
- Engine
- Max Power
- Seats

Target Variable:

- Selling Price

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS

## Machine Learning Algorithm

- Random Forest Regressor

## Model Performance

- R² Score: **0.9401**
- RMSE: **212299.82**

## Project Structure

```
CarDekho-Car-Price-Prediction/
│
├── app.py
├── model_training.py
├── best_model.pkl
├── requirements.txt
├── dataset/
├── templates/
├── static/
└── README.md
```

## How to Run

1. Install the dependencies:

```
pip install -r requirements.txt
```

2. Run:

```
python app.py
```

3. Open:

```
http://127.0.0.1:5000
```

## Future Improvements

- Better user interface
- Dropdown menus populated dynamically from the dataset
- Deployment on Render or Railway
- Additional regression model comparisons

## Author

**Aswath S**
