# from flask import Flask, request, render_template
import numpy as np
import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictionConfig, PredictPipeline


st.title("Math Score Predictor")
st.markdown("### Please fill in the details below:")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ['female', 'male'])
    race_ethnicity = st.selectbox('Race/Ethnicity', 
                                  ['group A', 'group B', 'group C', 
                                   'group D', 'group E'])
    parental_level_of_education = st.selectbox('Parental Level of Education', 
                                               ["bachelor's degree", 'some college', 
                                                "master's degree", "associate's degree", 
                                                'high school', 'some high school'])
    lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
    test_preparation_course = st.selectbox("Test Preparation Course", ['none', 'completed'])
    reading_score = st.slider("Reading Score", 0, 100, 70)
    writing_score = st.slider("Writing Score", 0, 100, 70)

    submit = st.form_submit_button("Predict Math Score")

if submit:
    data = CustomData(gender, race_ethnicity, 
                      parental_level_of_education, 
                      lunch, test_preparation_course, 
                      reading_score, writing_score)
    input_df = data.get_dataframe()
    # print(input_df)

    predconfig = PredictionConfig()
    predict_pipeline = PredictPipeline(predconfig.preprocessor_path, predconfig.model_path)
    
    prediction = predict_pipeline.predict(input_df)
    
    st.success(f"Predicted Math Score: **{np.round(prediction[0],2)}**")


