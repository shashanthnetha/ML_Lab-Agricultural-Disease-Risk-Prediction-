
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("dataset.csv")

# 2. Separate inputs and target
X = df.drop("risk", axis=1)
y = df["risk"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the Logistic Regression model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict the test data
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Results")
print("---------------------------")
print("Actual values:    ", y_test.tolist())
print("Predicted values: ", y_pred.tolist())
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
