
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("dataset.csv")

# Separate inputs and target
X = df.drop("risk", axis=1)
y = df["risk"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the SVM model
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf")
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate the model
print("SVM Results")
print("-----------")
print("Actual values:   ", y_test.tolist())
print("Predicted values:", y_pred.tolist())

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

