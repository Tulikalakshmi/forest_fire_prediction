import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv("Forest_fire.csv")

# Data cleaning
# Drop rows with missing values
data.dropna(inplace=True)

# Convert necessary columns to numeric (excluding Area column)
data['Oxygen'] = pd.to_numeric(data['Oxygen'], errors='coerce')
data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')
data['Humidity'] = pd.to_numeric(data['Humidity'], errors='coerce')
data['Fire Occurrence'] = pd.to_numeric(data['Fire Occurrence'], errors='coerce')

# Drop rows with any non-convertible values
data.dropna(inplace=True)

# Extract features and target variable
X = data[['Oxygen', 'Temperature', 'Humidity']]
y = data['Fire Occurrence']

# Convert to numpy arrays
X = X.values
y = y.astype('int').values

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Test the model (optional: you can print accuracy or other metrics)
# accuracy = log_reg.score(X_test, y_test)
# print(f"Model Accuracy: {accuracy:.2f}")

# Save the model using pickle
pickle.dump(log_reg, open('model.pkl', 'wb'))
