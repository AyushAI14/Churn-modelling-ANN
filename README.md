# Churn Prediction App

This project is a machine learning-powered web application for predicting whether a bank customer will churn (leave the service). The app is built with Streamlit and uses a trained deep learning model.

## Features

* Takes user inputs such as age, geography, credit score, balance, and account details
* Encodes and scales features to match model training
* Loads trained TensorFlow model and preprocessing artifacts
* Predicts churn probability and shows result to user

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Streamlit
* Pandas / NumPy
* Joblib

## Project Structure

```
├── model/
│   └── model_1.h5
├── notebook/
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── model_columns.pkl
├── app.py
└── README.md
```

## How It Works

1. User inputs customer information through the Streamlit UI
2. Categorical features are encoded (gender and geography)
3. Numerical features are scaled using the saved scaler
4. Model predicts the probability of churn
5. Output message indicates whether the customer is likely to churn

## Installation

### Clone the repository

```
git clone https://github.com/AyushAI14/Churn-modelling-ANN.git 
cd Churn-modelling-ANN
```

### Create and activate environment

```
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### Install dependencies

```
pip install -r requirements.txt
```

## Run the App

```
streamlit run app.py
```

## Model Files

Ensure the following files are present:

* `model/model_1.h5`
* `notebook/scaler.pkl`
* `notebook/encoder.pkl`
* `notebook/model_columns.pkl`
