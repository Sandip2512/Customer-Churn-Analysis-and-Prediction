import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import os
from user_management import authenticate, add_user
from preprocessing import preprocess

# Global model variable
model = None

# Define main application
def main():
    global model
    st.title('Telco Customer Churn Prediction App')

    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, 'App.jpg')

    # Check if the file exists before loading
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='App Image')
    else:
        st.info("App Image not found.")

    # Load the model if not already loaded
    model_path = os.path.join(current_dir, 'notebook', 'model.sav')
    if model is None:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.write("Model loaded successfully!")
        else:
            st.error("Model file not found. Please ensure the model path is correct.")

    # Provide option to train and save the model if missing
    if model is None:
        st.warning("No model found. Please train a new model.")
        if st.button("Train Model"):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            # Example data for demonstration
            df = pd.DataFrame({
                'SeniorCitizen': [0, 1, 0, 1],
                'tenure': [10, 5, 20, 30],
                'MonthlyCharges': [70, 80, 60, 90],
                'TotalCharges': [700, 400, 1200, 2700],
                'target': [1, 0, 0, 1]
            })
            X = df.drop('target', axis=1)
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            st.success(f"Model trained with accuracy: {accuracy * 100:.2f}%")
            # Save the trained model
            os.makedirs(os.path.join(current_dir, 'notebook'), exist_ok=True)
            model_save_path = os.path.join(current_dir, 'notebook', 'model.sav')
            joblib.dump(model, model_save_path)
            st.success(f"Model saved successfully at {model_save_path}")

    # Application sidebar
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
    )
    st.sidebar.info('This app predicts Customer Churn.')
    if os.path.exists(image_path):
        st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")

        # Demographic Data Input
        st.subheader("Demographic Data")
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))

        # Payment Data Input
        st.subheader("Payment Data")
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=1)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        payment_method = st.selectbox('Payment Method',
                                       ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
        monthlycharges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=50.0)
        totalcharges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=500.0)

        # Services Data Input
        st.subheader("Services Signed Up For")
        mutliplelines = st.selectbox("Multiple Lines", ('Yes', 'No', 'No phone service'))
        phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
        internetservice = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Online Security", ('Yes', 'No', 'No internet service'))
        onlinebackup = st.selectbox("Online Backup", ('Yes', 'No', 'No internet service'))
        techsupport = st.selectbox("Technology Support", ('Yes', 'No', 'No internet service'))
        streamingtv = st.selectbox("Streaming TV", ('Yes', 'No', 'No internet service'))
        streamingmovies = st.selectbox("Streaming Movies", ('Yes', 'No', 'No internet service'))

        # Organize user inputs into a dictionary
        data = {
            'SeniorCitizen': seniorcitizen,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phoneservice,
            'MultipleLines': mutliplelines,
            'InternetService': internetservice,
            'OnlineSecurity': onlinesecurity,
            'OnlineBackup': onlinebackup,
            'TechSupport': techsupport,
            'StreamingTV': streamingtv,
            'StreamingMovies': streamingmovies,
            'Contract': contract,
            'PaperlessBilling': paperlessbilling,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthlycharges,
            'TotalCharges': totalcharges
        }
        features_df = pd.DataFrame([data])
        st.write('Overview of input data:')
        st.dataframe(features_df)

        # Preprocess and Predict
        preprocess_df = preprocess(features_df, 'Online')
        try:
            prediction = model.predict(preprocess_df)
            prediction_proba = model.predict_proba(preprocess_df)[0][1]

            if st.button('Predict'):
                if prediction == 1:
                    st.warning(f'Yes, the customer will terminate the service. Probability: {prediction_proba:.2f}')
                else:
                    st.success(f'No, the customer is happy with Telco Services. Probability: {1 - prediction_proba:.2f}')
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.subheader("Dataset Upload")
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write(data.head())
                preprocess_df = preprocess(data, "Batch")
                if st.button('Predict'):
                    prediction = model.predict(preprocess_df)
                    prediction_proba = model.predict_proba(preprocess_df)[:, 1]
                    prediction_df = pd.DataFrame({
                        "Predictions": prediction,
                        "Probability": prediction_proba
                    })
                    prediction_df["Predictions"] = prediction_df["Predictions"].replace({1: 'Customer will churn', 0: 'Customer is happy'})
                    st.subheader('Prediction Results')
                    st.write(prediction_df)
            except Exception as e:
                st.error(f"Error processing the file: {e}")

# Registration Functionality
def register():
    st.title("User Registration")
    new_username = st.text_input("Enter a username")
    new_password = st.text_input("Enter a password", type="password")

    if st.button("Register"):
        if new_username and new_password:
            success = add_user(new_username, new_password)
            if success:
                st.success("You have successfully registered!")
                st.session_state['registered'] = True
                st.session_state['is_registering'] = False
            else:
                st.warning("Username already exists. Please choose a different one.")
        else:
            st.warning("Please fill out all fields.")

    if st.button("Back to Login"):
        st.session_state['is_registering'] = False

# Login Functionality
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Logged in successfully!")
            main()
        else:
            st.error("Username or password is incorrect")

    if st.button("Register here"):
        st.session_state['is_registering'] = True

if __name__ == '__main__':
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'is_registering' not in st.session_state:
        st.session_state['is_registering'] = False
    if 'registered' not in st.session_state:
        st.session_state['registered'] = False

    if st.session_state['is_registering']:
        register()
    elif st.session_state['logged_in']:
        main()
    elif st.session_state['registered']:
        st.session_state['registered'] = False
        login()
    else:
        login()
