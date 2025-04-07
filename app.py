
import streamlit as st
import pandas as pd
import joblib

# Load saved model and label encoders
model = joblib.load("best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Car Evaluation Predictor")

def user_input_features():
    buying = st.selectbox("Buying Price", label_encoders['buying'].classes_)
    maint = st.selectbox("Maintenance Cost", label_encoders['maint'].classes_)
    doors = st.selectbox("Number of Doors", label_encoders['doors'].classes_)
    persons = st.selectbox("Capacity (persons)", label_encoders['persons'].classes_)
    lug_boot = st.selectbox("Luggage Boot Size", label_encoders['lug_boot'].classes_)
    safety = st.selectbox("Safety Level", label_encoders['safety'].classes_)

    data = {
        'buying': label_encoders['buying'].transform([buying])[0],
        'maint': label_encoders['maint'].transform([maint])[0],
        'doors': label_encoders['doors'].transform([doors])[0],
        'persons': label_encoders['persons'].transform([persons])[0],
        'lug_boot': label_encoders['lug_boot'].transform([lug_boot])[0],
        'safety': label_encoders['safety'].transform([safety])[0]
    }

    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    target_label = label_encoders['class'].inverse_transform([prediction])[0]
    st.success(f"The predicted car evaluation is: **{target_label}**")
