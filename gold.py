import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import streamlit as st 

from PIL import Image 


df= pd.read_csv('C:\\Users\\91700\\OneDrive\\Desktop\\gold_price_prediction\\gld_price_data.csv')

x=df.drop(['Date','GLD'],axis=1)
y=df['GLD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

reg =RandomForestRegressor()
reg.fit(x_train,y_train)
pred=reg.predict(x_test)
score = r2_score(y_test,pred)


st.title('Gold Price Prediction')
# img=Image.open('depositphotos_19492613-stock-photo-gold-ingots.jpg')
st.image('pexels-pixabay-47047.jpg',caption='gold_image')
# st.image(img,width=200,use_coloumn_width=True)
st.subheader('Using randomforestregressor')
st.write(df)
st.subheader('Model performance')
st.write(score)


