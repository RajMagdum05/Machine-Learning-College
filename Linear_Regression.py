import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Load Boston Housing dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# 2. Keep only numerical columns (already all numerical here)
df_num = df.dropna()

# Target column: 'medv' (Median value of owner-occupied homes in $1000s)
X = df_num.drop('medv', axis=1)
y = df_num['medv']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 5))

# Train Plot
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='blue', label='Actual Price (Train)')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', label='Predicted Line')
plt.title('Training Set: Actual vs Predicted')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)

# Test Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='green', label='Actual Price (Test)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Predicted Line')
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
