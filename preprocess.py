import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('C:/Users/Salee/PycharmProjects/TeamProject/heart.csv')

# Fill missing values
df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].mean())

# Map categorical to numerical
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['ChestPainType'] = df['ChestPainType'].map({'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4})
df['RestingECG'] = df['RestingECG'].map({'Normal': 1, 'ST': 2, 'LVH': 3})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
df['ST_Slope'] = df['ST_Slope'].map({'Up': 1, 'Flat': 0, 'Down': 2})

# Define features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Scale numerical features only
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression with class_weight='balanced'
model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, 'logistic_regression_model.pkl')
print("Saved logistic_regression_model.pkl")
joblib.dump(scaler, 'standard_scaler.pkl')
print("Saved standard_scaler.pkl")
