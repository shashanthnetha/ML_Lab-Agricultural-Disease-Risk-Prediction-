
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(
    page_title="Agricultural Disease Risk Prediction",
    page_icon="🌱"
)

st.title("🌱 Agricultural Disease Risk Prediction")
st.write("Predict crop disease risk using Machine Learning.")

# Load dataset
df = pd.read_csv("dataset.csv")

X = df.drop("risk", axis=1)
y = df["risk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create models
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

# Train all models
for model in models.values():
    model.fit(X_train, y_train)

# Model selection
selected_model = st.selectbox(
    "Choose a Machine Learning Algorithm",
    list(models.keys())
)

st.subheader("Enter Crop Conditions")

temperature = st.slider(
    "Temperature (°C)", 0, 50, 32
)

humidity = st.slider(
    "Humidity (%)", 0, 100, 85
)

rainfall = st.slider(
    "Rainfall (mm)", 0, 100, 20
)

soil_moisture = st.slider(
    "Soil Moisture (%)", 0, 100, 75
)

leaf_spots = st.selectbox(
    "Are leaf spots observed?",
    ["No", "Yes"]
)

# Convert inputs into model format
input_data = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
    "soil_moisture": soil_moisture,
    "leaf_spots": 1 if leaf_spots == "Yes" else 0
}])

# Prediction button
if st.button("Predict Disease Risk"):
    model = models[selected_model]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "High":
        st.error("Predicted Risk: HIGH")
    elif prediction == "Medium":
        st.warning("Predicted Risk: MEDIUM")
    else:
        st.success("Predicted Risk: LOW")

    st.write("Selected Model:", selected_model)
    st.write("Input Conditions:")
    st.dataframe(input_data, hide_index=True)

    st.caption(
        "Educational demonstration using a small synthetic dataset. "
        "This is not a validated agricultural diagnosis."
    )

# Model comparison
st.divider()
st.subheader("Model Accuracy Comparison")

results = []

for name, model in models.items():
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        )
    })

results_df = pd.DataFrame(results)

st.bar_chart(
    results_df.set_index("Model")["Accuracy (%)"]
)

st.dataframe(results_df, hide_index=True)
