
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load data
df = pd.read_csv("dataset.csv")

X = df.drop("risk", axis=1)
y = df["risk"]

# Use the same split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf")
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print(
        f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix (Low, Medium, High):")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["Low", "Medium", "High"]
    ))
