# 🎓 Student Placement Predictor

A Machine Learning web application that predicts whether a student is likely to get placed based on their **CGPA and IQ** using Logistic Regression. The application is built using Python and deployed with Streamlit.

## 🚀 Live Demo

[Click here to try the Student Placement Predictor](https://placement-prediction-app-dvsqw9dqbhvcsudzoho7se.streamlit.app/)

---

## 📸 Application Screenshot

![Placement Predictor](Placement-Predictor%20Screenshot.png)

---

## 📌 Project Overview

The Student Placement Predictor uses a Logistic Regression classification model to estimate student placement outcomes based on two academic features: CGPA and IQ.
The project follows an end-to-end Machine Learning workflow, including data preprocessing, exploratory data analysis, feature scaling, model training, evaluation, and deployment through an interactive Streamlit application.

## 🎯 Project Objectives

- Analyze student CGPA and IQ distributions.
- Explore the relationship between academic performance and placement outcomes.
- Build a Logistic Regression classification model.
- Evaluate model performance using classification metrics.
- Develop an interactive web application for placement prediction.

## 📂 Dataset Information

The project uses a student placement dataset (`placement.csv`).

| Column | Description |
|---|---|
| `cgpa` | Student's Cumulative Grade Point Average |
| `iq` | Student's Intelligence Quotient score |
| `placement` | Placement outcome (1 = Placed, 0 = Not Placed) |

The dataset may also contain an identifier column, which is excluded from the model.

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset structure and summary statistics.
- Missing value analysis.
- Distribution of placed and non-placed students.
- CGPA distribution.
- IQ distribution.
- Relationship between CGPA and IQ, categorized by placement status.

## ⚙️ Data Preprocessing

- Removed the unnecessary identifier column.
- Selected CGPA and IQ as independent features.
- Selected placement as the target variable.
- Split the dataset into training and testing sets (80:20).
- Used stratified sampling to maintain the placement class distribution.
- Applied StandardScaler for feature scaling through a Machine Learning pipeline.

## 🤖 Machine Learning Model

**Algorithm:** Logistic Regression

The model was implemented using a Scikit-learn Pipeline consisting of:

1. StandardScaler – Standardizes CGPA and IQ features.
2. Logistic Regression – Performs binary classification to predict placement outcomes.

**Train-Test Split:**
- Training Data: 80%
- Testing Data: 20%
- Random State: 42

The trained pipeline is saved as `placement_model.pkl` using Joblib and reused by the Streamlit application.

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help assess the model's ability to correctly identify placed and non-placed students.

## 🌐 Streamlit Application Features

The application provides three sections:

### 1. Placement Prediction
- Enter student CGPA and IQ.
- Predict whether the student is likely to be placed.
- View the estimated placement probability.

### 2. Model Performance
- View testing accuracy.
- Examine the classification report.
- Visualize the confusion matrix.

### 3. Dataset Insights
- View total students, placed students, and non-placed students.
- Analyze the overall placement rate.
- Explore placement distribution.
- Visualize CGPA and IQ distributions.
- Examine the relationship between CGPA and IQ.
- View dataset preview and statistical summary.
- Compare average CGPA and IQ across placement groups.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine Learning and model evaluation
- **Joblib** – Model serialization
- **Streamlit** – Web application development
- **Git & GitHub** – Version control and project hosting

## 📁 Project Structure

```text
Placement-Predictor/
│
├── Placement-Predictor.py
├── app.py
├── placement.csv
├── placement_model.pkl
├── requirements.txt
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SM41b/Placement-Predictor.git
```

### 2. Navigate to the Project Directory

```bash
cd Placement-Predictor
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```powershell
.\venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes. Placement predictions are based only on CGPA and IQ and should not be considered a definitive assessment of a student's employability or actual placement outcome.
