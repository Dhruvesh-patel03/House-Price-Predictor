# House-Price-Predictor
This repository features a linear regression model for predicting house prices based on factors like square footage, bedrooms, and bathrooms. Built with Python using scikit-learn, pandas, and matplotlib, it provides insights into real estate pricing and visualizes prediction accuracy, aiding understanding of how home features influence valuation.

from google.colab import files
uploaded = files.upload()  # This opens a file upload dialog

import pandas as pd
data = pd.read_csv('train.csv')  # Adjust the filename if it’s different
print(data.head())
print(data.columns)  # Checks for exact column names
print(data.isnull().sum())  # Checks for missing values in each column

# Defined features (X) and target variable (y)
data = data.dropna(subset=['BsmtFullBath', 'BsmtHalfBath', 'FullBath' , 'HalfBath' , 'LotArea', 'SalePrice'])
X = data[['BsmtFullBath', 'BsmtHalfBath', 'FullBath' , 'HalfBath' , 'LotArea']]
y = data['SalePrice'] 

print("X shape:", X.shape)  # Should be (num_rows, 3)
print("y shape:", y.shape)  # Should be (num_rows,)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# For Predictions
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

import matplotlib.pyplot as plt

# For Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linestyle='--', linewidth=2)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Price")
plt.show()
