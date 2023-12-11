import streamlit as st
import pickle
import pandas as pd
import os
import sklearn

st.header("Detection of Heart Disease's presence")
algorithm = st.selectbox('Algorithm to be performed:',('Support Vector Machine','K-Nearest Neighbours','Logistic Regression','Naive Bayes','Decision Tree'))

st.write("Enter the values for the following attribute:")
chestpain = st.selectbox("Chest pain type:",('typical angina pain','atypical angina pain','non-anginal pain','asymptomatic pain'))
restingBP = st.number_input("Resting blood pressure:", min_value = 94, max_value = 200)
restingelectro = st.selectbox("Resting electro cardiogram results:",('normal','having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)','showing probable or definite left ventricular hypertrophy by Estes criteria)'))
slope = st.selectbox("Slope of the peak exercise ST segment:",('upsloping','flat','downsloping'))
noofmajorvessels = st.number_input("Number of major vessels:", min_value = 0, max_value = 3)

model = {'Support Vector Machine': 'SVM', 'K-Nearest Neighbours': 'KNN', 'Logistic Regression': 'LR', 'Naive Bayes': 'NB', 'Decision Tree': 'Dtree'}
os.chdir("/mount/src/detection_of_heart_disease/Models/")

with open(model[algorithm]+'.pkl', 'rb') as file:  
    classifier = pickle.load(file)

cp = {'typical angina pain': 0,'atypical angina pain': 1,'non-anginal pain': 2,'asymptomatic pain': 3}
re = {'normal': 0,'having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)': 1,'showing probable or definite left ventricular hypertrophy by Estes criteria)': 2}
slp = {'upsloping': 1,'flat': 2,'downsloping': 3}

X_test = [[cp[chestpain],restingBP,re[restingelectro],slp[slope],noofmajorvessels]]

y_predict = classifier.predict(X_test)

if y_predict == 0: result = 'Absence of Heart Disease'
else: result = 'Presence of Heart Disease'

st.subheader(result)
