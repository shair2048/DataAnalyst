import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix

def execute_model(csv_dt):
    target_variable = st.selectbox("Select Target Variable", csv_dt.columns)
    input_variables = st.multiselect("Select Input Variables", csv_dt.columns)

    le = LabelEncoder()

    # Store original labels
    original_labels = {}

    # if input variable is categorical convert to numerical
    for column in input_variables:
        if csv_dt[column].dtype == "object":
            original_labels[column] = csv_dt[column].unique()
            csv_dt[column] = le.fit_transform(csv_dt[column])

    if csv_dt[target_variable].dtype == "object":
        original_labels[target_variable] = csv_dt[target_variable].unique()
        encoded_target = le.fit_transform(csv_dt[target_variable])

        # Convert continuous target variable into categorical
        bins = np.linspace(min(encoded_target), max(encoded_target), 4)
        encoded_target = np.digitize(encoded_target, bins)

    else:
        original_labels[target_variable] = None
        encoded_target = csv_dt[target_variable]

    execute_button = st.button("Execute Model")
    tabs = st.tabs(["Linear Regression", "Logistic Regression", "KNN"])
    
    if execute_button:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            csv_dt[input_variables], encoded_target, test_size=0.2, random_state=42
        )

        with tabs[0]:  # Linear Regression tab
            st.write("## Linear Regression")

            # Create and fit the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            r2 = r2_score(y_test, y_pred)
            st.write("R^2 Score:", r2)

            # Plotting scatter plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title('Linear Regression: True vs Predicted Values')
            st.pyplot(fig)

        with tabs[1]:  # Logistic Regression tab
            st.write("## Logistic Regression")

            # Create and fit the model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Accuracy Score:", accuracy)

            # Plotting confusion matrix
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Logistic Regression: Confusion Matrix')
            # Restore original labels
            if original_labels[target_variable] is not None:
                ax.set_xticks(range(len(original_labels[target_variable])))
                ax.set_xticklabels(original_labels[target_variable], rotation=45)
            st.pyplot(fig)

        with tabs[2]:  # KNN tab
            st.write("## K-Nearest Neighbors")

            # Create and fit the model
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Accuracy Score:", accuracy)

            # Plotting confusion matrix
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('KNN: Confusion Matrix')
            # Restore original labels
            if original_labels[target_variable] is not None:
                ax.set_xticks(range(len(original_labels[target_variable])))
                ax.set_xticklabels(original_labels[target_variable], rotation=45)
            st.pyplot(fig)

# Example usage:
# execute_model(csv_dt)
