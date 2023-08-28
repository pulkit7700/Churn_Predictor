

# Importing the Required Modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import joblib
import numpy as np 
import pandas as pd

# Importing the Model trained and its preproceessor to be used in the model
model = load_model('Models/Churn-Model-ANN-1693242575')
preprocessor = joblib.load("dep_processor.jb")

# Creating Input Field to be unputted in the model

st.title("Churn Predictor") # 

id = st.text_input("Enter Customer Id")
name = st.text_input("Name of the Customer")
age = st.number_input("Enter the Age")
gender = st.selectbox("Enter the Gender of the Person", ["Male", 'Female'])
location = st.selectbox("Enter the Location of the Person", ['Houston', 'Los Angeles', "Miami", "Chicago", 'New York'])
slm = st.number_input("Enter Subscription Length Month")
monthly_bill = st.number_input("Enter bill")
total_usage = st.number_input('Enter Total Usage')

# Hellper Columns which would be used in the helper function of the model
columns = ['Age', 'Gender',	'Location',	'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']


# Helper Function to predict the model
def predict():
    row = np.array([age, gender, location, slm, monthly_bill, total_usage]) 
    X = pd.DataFrame([row], columns = columns)
    X = preprocessor.transform(X)
    prediction = model.predict(X)[0]
    if prediction > 0.5:
        prediction = "Person is About to Leave the Business"
    else:
        prediction = "Person Will Stay in the Business"
    st.success(prediction)
    return prediction

if st.button('Price Prediction', on_click=predict):
    predict()
