import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# 1. LOAD THE DATASET
# Ensure 'car data.csv' is in your project folder
df = pd.read_csv('car data.csv')

# 2. FEATURE ENGINEERING
# The 'Year' of purchase is less useful than the 'Age' of the car.
# We calculate Age by subtracting the purchase year from the current year.
current_year = 2024
df['Age'] = current_year - df['Year']

# Drop columns that aren't useful for numerical prediction
# Car_Name has too many unique values, and Year is now replaced by Age.
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# 3. DATA PREPROCESSING (ENCODING)
# Machine Learning models only understand numbers.
# We convert categorical text (Fuel_Type, Transmission, etc.) into numerical values.
# pd.get_dummies creates "dummy" columns (One-Hot Encoding).
df = pd.get_dummies(df, drop_first=True)



# 4. SPLITTING THE DATA
# X contains the independent variables (Features like kms driven, age, etc.)
# Y contains the dependent variable (Target: Selling_Price)
X = df.drop('Selling_Price', axis=1)
Y = df['Selling_Price']

# Split data into Training set (80%) and Testing set (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5. MODEL BUILDING
# Using RandomForestRegressor: An ensemble learning method that uses multiple 
# decision trees to provide highly accurate regression results.
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)



# 6. MODEL EVALUATION
# Predict values for the training data to check how well the model learned
training_prediction = model.predict(X_train)
r2_score = metrics.r2_score(Y_train, training_prediction)
print(f"Model Accuracy (R-squared Score): {r2_score * 100:.2f}%")

# 7. VISUALIZING RESULTS
# Plotting Actual vs Predicted prices to see the model performance visually
test_prediction = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, test_prediction, alpha=0.7, color='teal')
# A 45-degree line represents perfect prediction
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', lw=2)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()



# 8. MAKING A PREDICTION
# Testing the model with a single piece of data from the test set
print("\n--- Manual Check ---")
print(f"Actual Price of car: {Y_test.iloc[0]}")
print(f"Model's Predicted Price: {test_prediction[0]:.2f}")
