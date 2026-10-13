# 🌱 Agricultural Disease Risk Prediction

A beginner-friendly Machine Learning project that classifies agricultural disease risk into three categories: **Low, Medium, and High**.

The project implements four classification algorithms and compares their performance using a synthetic dataset of 5,000 records.

## 🚀 Features

- Generates a dataset containing 5,000 synthetic records.
- Prepares training and testing datasets.
- Implements four Machine Learning algorithms.
- Compares model accuracy using a bar chart.
- Evaluates models using precision, recall, F1-score, and confusion matrices.
- Provides an interactive web interface using Streamlit.

## 🤖 Machine Learning Algorithms

1. **Logistic Regression** — Classifies observations into risk categories.
2. **Decision Tree Classifier** — Uses decision rules to classify risk.
3. **Random Forest Classifier** — Combines multiple decision trees.
4. **Support Vector Machine (SVM)** — Separates classes using decision boundaries.

## 📊 Dataset Information

The dataset contains 5,000 synthetically generated records.

| Feature | Description |
|---|---|
| temperature | Temperature in degrees Celsius |
| humidity | Relative humidity percentage |
| rainfall | Rainfall in millimeters |
| soil_moisture | Soil moisture percentage |
| leaf_spots | 0 = No, 1 = Yes |
| risk | Low, Medium, or High |

**Important:** This dataset is synthetic. The risk labels are generated using an illustrative scoring formula and do not represent verified agricultural disease observations. The model results demonstrate classification techniques, not validated real-world disease prediction.

## 📈 Model Performance

The following results were obtained using an 80/20 train-test split with 1,000 test records.

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 68.60% |
| Decision Tree | 68.80% |
| Random Forest | 76.60% |
| SVM | 74.20% |

Random Forest achieved the highest accuracy in this experiment.

These results measure agreement with synthetic labels and should not be interpreted as real-world agricultural disease prediction accuracy.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## 📁 Project Structure

```text
agricultural-disease-prediction/
├── dataset.csv
├── generate_data.py
├── prepare_data.py
├── logistic_model.py
├── decision_tree.py
├── random_forest.py
├── svm_model.py
├── compare_models.py
├── evaluate_models.py
├── app.py
├── model_comparison.png
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/agricultural-disease-prediction.git
cd agricultural-disease-prediction
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## ▶️ Run the Project

### Launch the web application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

### Compare all four models

```bash
python compare_models.py
```

### View detailed evaluation metrics

```bash
python evaluate_models.py
```

### Generate a new dataset

```bash
python generate_data.py
```

**Warning:** Running the dataset generation script overwrites `dataset.csv` with a new synthetic dataset.

## 🎯 Project Objectives

- Understand supervised Machine Learning classification.
- Implement and compare four classification algorithms.
- Learn data preparation and train-test splitting.
- Evaluate classification models using standard metrics.
- Build a simple interactive ML application.

## 🔮 Future Improvements

- Use real agricultural disease and weather datasets.
- Evaluate performance using cross-validation.
- Improve model tuning and feature selection.
- Add crop-specific disease risk prediction.
- Deploy the application for demonstration.

## 👨‍🎓 Academic Use

This project is intended for educational purposes and Machine Learning laboratory demonstrations.

It is not a substitute for professional agricultural diagnosis or field-validated disease forecasting.
