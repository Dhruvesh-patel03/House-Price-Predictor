# House-Price-Predictor
This repository features a linear regression model for predicting house prices based on factors like square footage, bedrooms, and bathrooms. Built with Python using scikit-learn, pandas, and matplotlib, it provides insights into real estate pricing and visualizes prediction accuracy, aiding understanding of how home features influence valuation.


### Linear Regression Model for House Price Prediction

This project implements a linear regression model to predict house prices based on several features related to the house's square footage, the number of bathrooms, and other relevant attributes. The model is built using Python's `scikit-learn` library and evaluates its performance with metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE).

The code reads a dataset (`train.csv`) containing information on house attributes and prices. After checking for null values and filtering out incomplete entries, the dataset is split into training and test sets. The features used in this model include `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, and `LotArea`, with the target variable being `SalePrice`.

The model is trained on the training set, and predictions are generated on the test set to evaluate performance. Additionally, a scatter plot visualizes the model's accuracy by plotting actual sale prices against predicted values, with a reference line indicating ideal predictions.

#### Requirements
- Python
- `pandas`, `scikit-learn`, and `matplotlib` for data manipulation, model training, and visualization

#### Usage
To run the code, ensure `train.csv` is in your working directory, and then execute each cell in sequence.
