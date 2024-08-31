import streamlit as st
import pandas as pd
import sklearn
import joblib
st.title("Insurance Prediction")
model =joblib.load("reg_model.joblib")
enc = joblib.load('encoder.joblib')
le_sex = joblib.load('le_sex.joblib')
le_smk = joblib.load('le_smk.joblib')

age = st.number_input('Enter age', min_value = 18)

sex = st.selectbox('Enter sex', ['male', 'Female']) 

bmi = st.number_input('Enter BMI', min_value = 15)

children = st.number_input('Number of children', min_value= 0)

smoker = st.selectbox('Smoker',['yes', 'no'])

region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])



data = {
        "age": age,
        "sex" : sex,
        "bmi" : bmi,
        "children": children,
        "smoker": smoker,
        "region": region}

df = pd.DataFrame(data, index=[0])

one_hot = enc.transform(df[['region']]).toarray()
df[["northeast","northwest","southeast","southwest"]] = one_hot
df['sex'] = le_sex.transform(df[['sex']])
df['smoker'] = le_smk.transform(df[['smoker']])
df = df.drop(columns='region')

button = st.button('Predict')


if button == True:
    prediction = model.predict(df.head(1))
    if prediction < 0:
        st.info(f'${0:0,.2f}')
    st.info(f'${prediction[0]:0,.2f}')