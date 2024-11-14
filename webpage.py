import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

scaler = pickle.load(open('scaler.sav', 'rb'))
parkinsons_model = pickle.load(open('model.sav', 'rb'))

df=pd.read_csv('parkinsons.csv')
feature_columns = ['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
           'MDVP:PPQ', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 
           'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 
           'DFA', 'D2']
features = df[feature_columns]
# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")
with st.sidebar:
    selected = option_menu('Parkinsons Disease Prediction System',
                           ['Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['person'],
                           default_index=0)
st.header("Enter the values")

col1, col2, col3 = st.columns(3)




with col1:
    MDVP_Fo_Hz = st.text_input('MDVP:\nFo(Hz)')
    MDVP_Fhi_Hz = st.text_input('MDVP:\nFhi(Hz)')
    MDVP_Flo_Hz = st.text_input('MDVP:\nFlo(Hz)')
    MDVP_Jitter = st.text_input('MDVP:\nJitter(%)')
    MDVP_Jitter_Abs = st.text_input('MDVP:\nJitter(Abs)')
    MDVP_RAP = st.text_input('MDVP:\nRAP')
    MDVP_PPQ = st.text_input('MDVP:\nPPQ')
    

with col2:
    Jitter_DDP = st.text_input('Jitter:\nDDP')
    MDVP_Shimmer = st.text_input('MDVP:\nShimmer')
    MDVP_Shimmer_dB = st.text_input('MDVP:\nShimmer(dB)')
    Shimmer_APQ3 = st.text_input('Shimmer:\nAPQ3')
    Shimmer_APQ5 = st.text_input('Shimmer:\nAPQ5')
    MDVP_APQ = st.text_input(' MDVP:\nAPQ')
    Shimmer_DDA = st.text_input('Shimmer:\nDDA')
    

with col3:
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    D2 = st.text_input('D2')
    spread1 = st.text_input('spread1')
    sprea2=st.text_input('spread2')
    PPE = st.text_input('PPE')

parkinsons_diagnosis = ''


def cap_row_with_df_iqr(input_row_df, df, factor=1.5):
    capped_row = input_row_df.copy()
    for col in input_row_df.columns:
        # Calculate IQR from the original DataFrame
        Q1 = np.quantile(df[col], 0.25)
        Q3 = np.quantile(df[col], 0.75)
        col_iqr = Q3 - Q1
            
        # Calculate lower and upper bounds based on IQR and factor
        lower_bound = Q1 - factor * col_iqr
        upper_bound = Q3 + factor * col_iqr

        # Cap the value if it's outside the bounds
        capped_row[col] = np.clip(input_row_df[col], lower_bound, upper_bound)
    return capped_row


# Creating a button for Prediction
if st.button("Predict Parkinson's"):
    
        # Convert input to float inside the button click
    user_input = [MDVP_Fo_Hz,MDVP_Flo_Hz, MDVP_Jitter,MDVP_Jitter_Abs,MDVP_PPQ,
                  MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,
                  Shimmer_DDA,NHR, HNR, RPDE, DFA, D2]
    user_input_float = [float(value) for value in user_input]

    input_row_df = pd.DataFrame([user_input_float], columns=feature_columns)
    
    capped_row_df=cap_row_with_df_iqr(input_row_df, features, factor=1.5)
        # Scale input
    scaled_input = scaler.transform(capped_row_df)

        # Make prediction
    parkinsons_prediction = parkinsons_model.predict(scaled_input)
    

        # Show result
    if parkinsons_prediction[0] == 1:
        st.write('<h2 style="color: red;">The output is Positive for Parkinsons Disease</h2>', unsafe_allow_html=True)
        st.write("Recommendations:")
        st.write("- Consult a neurologist for further evaluation.")
        st.write("- Consider undergoing additional diagnostic tests.")
        st.write("- Begin appropriate treatment and therapy.")
    else:
        st.write('<h2 style="color: green;">The output is Negative for Parkinsons Disease</h2>', unsafe_allow_html=True)
        st.write("Recommendations:")
        st.write("- Continue monitoring for any changes in vocal features.")
        st.write("- Maintain a healthy lifestyle with regular exercise and balanced diet.")
        st.write("- Follow up with healthcare provider if any symptoms develop.")
        