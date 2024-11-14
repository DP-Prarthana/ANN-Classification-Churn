import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

##load the trained model
model = tf.keras.models.load_model('model.h5')

##load the encoders and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_goe.pkl','rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scalar.pkl','rb') as file:
    scaler = pickle.load(file)

##streamlit app
st.title('CUSTOMER CHURN PREDICTION')

##User input

geography = st.selectbox('Geography',label_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('EstimatedSalary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number Of Products',1,4)
has_cr_card = st.selectbox('Has Cr Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

##prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded = label_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded,columns= label_encoder_geo.get_feature_names_out(['Geography']))

##cocatination with OHE 
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis = 1)

input_data_scaled = scaler.transform(input_data)

#prediction churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability:{prediction_prob:.2f}')

if prediction_prob >0.5:
    st.write('THe customer is likely to churn')
else:
    st.write('THe is not customer is likely to churn')