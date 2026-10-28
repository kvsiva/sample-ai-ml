import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample customer dataset
data = {
    "tenure": [2, 36, 6, 48, 3, 24, 12, 60],
    "monthly_charges": [1200, 800, 1500, 700,
                        1300, 900, 1100, 600],
    "complaints": [5, 0, 3, 0, 4, 1, 2, 0],
    "churn": [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Input features and target
X = df[["tenure", "monthly_charges", "complaints"]]
y = df["churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Train ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Predict churn for a new customer
new_customer = pd.DataFrame({
    "tenure": [3],
    "monthly_charges": [1400],
    "complaints": [4]
})

churn_probability = model.predict_proba(
    new_customer
)[0][1]

print("Churn Probability:", churn_probability)

if churn_probability >= 0.70:
    print("High Risk: Contact Customer")
elif churn_probability >= 0.40:
    print("Medium Risk: Monitor Customer")
else:
    print("Low Risk: No Immediate Action")