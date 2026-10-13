
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("dataset.csv")

# Separate inputs and output
X = df.drop("risk", axis=1)
y = df["risk"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the results
print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nInput features:")
print(X.columns.tolist())

print("\nTraining risk categories:")
print(y_train.value_counts())

print("\nData preparation completed!")
