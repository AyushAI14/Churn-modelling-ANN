import streamlit as st
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
import joblib 
import tensorflow as tf 

# model loading 
model = tf.keras.models.load_model('model/model_1.h5')


scaler = joblib.load('notebook/scaler.pkl')

st.title("CHURN MODEL PREDICTOR")

# user input 
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
Gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

label_enc = joblib.load('notebook/encoder.pkl')
model_col = joblib.load('notebook/model_columns.pkl')

#input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender':[Gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#preprocessing the data
# Encode Gender
input_data['Gender'] = label_enc.transform(input_data['Gender'])

# One-hot encode Geography
geo_df = pd.get_dummies(input_data['Geography'], prefix='Geography', dtype=int)

print(geo_df)
# Drop original Geography and concat
input_data = input_data.drop(columns=['Geography'])
input_data = pd.concat([input_data, geo_df], axis=1)

# Reindex to match training columns
input_data = input_data.reindex(columns=model_col, fill_value=0)
print(input_data)
# Scale
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)[0][0]


st.write('Churn Prediction',prediction)

if prediction > 0.5:
    st.write('The Consumer is likely to Churn')
else :
        st.write('The Consumer will not Churn')