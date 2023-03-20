import pandas as pd
import numpy as np
import pickle 
from sklearn import *
import streamlit as st 

df = pickle.load(open('df.pkl', 'rb'))
model = pickle.load(open('lr.pkl','rb'))

st.title("Car Price Prediction")
st.header("Fill details to predict price")


# features = ['year', 'selling_price', 'km_driven', 'fuel', 'transmission', 'owner','company']
year = st.selectbox('year', [i for i in range(2021,2000,-1)])
km_driven = st.number_input("Enter Kilometer driven")
fuel = st.selectbox('fuel', df['fuel'].unique())
transmission = st.selectbox('transmission', df['transmission'].unique())
owner = st.selectbox('owner', df['owner'].unique())
company = st.selectbox('company', df['company'].unique())


if st.button("Predict Car price"):
    test_data = pd.DataFrame([[year,km_driven,fuel,transmission,owner,company]],columns=['year','km_driven','fuel','transmission','owner','company'])
    #test_data =test_data.reshape([1,7])
    #test_data = pd.DataFrame(test_data)

    st.success(model.predict(test_data)[0])
