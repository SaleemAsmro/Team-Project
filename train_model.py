import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the preprocessed dataset
df = pd.read_csv('C:/Users/Salee/PycharmProjects/TeamProject/heart_processed.csv')
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print("Report:\n", classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, 'logistic_regression_model.pkl')
