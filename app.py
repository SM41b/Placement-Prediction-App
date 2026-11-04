
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Predictor")

st.write(
    "Predict student placement outcomes using "
    "CGPA and IQ with Machine Learning."
)


# --------------------------------------------------
# LOAD DATASET AND TRAINED MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("placement.csv")


@st.cache_resource
def load_model():
    return joblib.load("placement_model.pkl")


try:
    df = load_data()
    model = load_model()

except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e.filename}. "
        "Make sure placement.csv and placement_model.pkl "
        "are in the project folder."
    )
    st.stop()


# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------

# Select the columns used during model training
required_columns = ["cgpa", "iq", "placement"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Dataset is missing columns: {missing_columns}")
    st.stop()

data = df[required_columns].dropna().copy()

X = data[["cgpa", "iq"]]
y = data["placement"]


# --------------------------------------------------
# TRAIN-TEST SPLIT FOR MODEL EVALUATION
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Placement Prediction",
        "Model Performance",
        "Dataset Insights"
    ]
)


# --------------------------------------------------
# PAGE 1: PLACEMENT PREDICTION
# --------------------------------------------------

if page == "Placement Prediction":

    st.header("Student Placement Prediction")

    st.write(
        "Enter the student's academic details "
        "to estimate their placement outcome."
    )

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider(
            "Enter CGPA",
            float(data["cgpa"].min()),
            float(data["cgpa"].max()),
            float(data["cgpa"].median()),
            step=0.1
        )

    with col2:
        iq = st.slider(
            "Enter IQ",
            int(data["iq"].min()),
            int(data["iq"].max()),
            int(data["iq"].median()),
            step=1
        )

    st.divider()

    if st.button("Predict Placement", type="primary"):

        input_data = pd.DataFrame(
            [[cgpa, iq]],
            columns=["cgpa", "iq"]
        )

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        positive_class_index = list(model.classes_).index(1)

        placement_probability = (
            probabilities[positive_class_index] * 100
        )

        if prediction == 1:
            st.success("The student is likely to be placed.")
        else:
            st.warning("The student is likely not to be placed.")

        st.metric(
            "Estimated Placement Probability",
            f"{placement_probability:.2f}%"
        )

        st.progress(
            int(round(placement_probability))
        )

        st.caption(
            "This is a machine-learning estimate, not a guarantee "
            "of actual placement."
        )


# --------------------------------------------------
# PAGE 2: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    col1.metric(
        "Testing Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Testing Samples",
        len(y_test)
    )

    st.divider()

    # Classification Report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Placement")
    ax.set_ylabel("Actual Placement")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# PAGE 3: DATASET INSIGHTS
# --------------------------------------------------

elif page == "Dataset Insights":

    st.header("Dataset Insights")

    total_students = len(data)

    placed_students = int((data["placement"] == 1).sum())

    not_placed_students = int((data["placement"] == 0).sum())

    placement_rate = (
        placed_students / total_students
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Placed Students", placed_students)
    col3.metric("Not Placed", not_placed_students)
    col4.metric("Placement Rate", f"{placement_rate:.2f}%")

    st.divider()

    # Placement Distribution
    st.subheader("Placement Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        data=data,
        x="placement",
        hue="placement",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title("Placed vs Not Placed Students")
    ax.set_xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
    ax.set_ylabel("Number of Students")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # CGPA and IQ distributions
    st.subheader("Student Academic Distributions")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.histplot(
            data=data,
            x="cgpa",
            kde=True,
            ax=ax
        )

        ax.set_title("CGPA Distribution")

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.histplot(
            data=data,
            x="iq",
            kde=True,
            ax=ax
        )

        ax.set_title("IQ Distribution")

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # CGPA vs IQ
    st.subheader("CGPA vs IQ by Placement Status")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=data,
        x="cgpa",
        y="iq",
        hue="placement",
        palette="Set1",
        s=80,
        ax=ax
    )

    ax.set_title("Relationship Between CGPA and IQ")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("IQ")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Dataset Preview
    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    # Statistical Summary
    st.subheader("Statistical Summary")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True
    )

    # Average CGPA and IQ by placement status
    st.subheader("Average CGPA and IQ by Placement Status")

    grouped_data = data.groupby("placement")[
        ["cgpa", "iq"]
    ].mean().round(2)

    grouped_data.index = [
        "Not Placed" if index == 0 else "Placed"
        for index in grouped_data.index
    ]

    st.dataframe(
        grouped_data,
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Placement Predictor | Machine Learning Project"
)
