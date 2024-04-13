import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pyarrow
import fastparquet
import os
import pickle
from utils import *
import streamlit_ext as ste

st.markdown("<h1 style='text-align: center; color: black;'>Child Mind Institute challenge - sleep detection</h1>", unsafe_allow_html=True)
#st.title("Child Mind Institute challenge - sleep detection")
# image titre
img_title = Image.open(os.path.join("input", "Health_theme.jpg"))
st.image(img_title, caption="Source:https://sleepopolis.com/wp-content/uploads/2022/06/WhatDoesApplesNewSleepAppDo_Header-1024x650.jpg")
st.markdown("*Detection of sleep onset and wake from wrist-worn accelerometer data*:watch:")

st.markdown('<div style="text-align: justify;">\
            The accelerometer data consists of 2 features, enmo and anglez.<br>\
            The time difference between records is 5 seconds.<br>\
            <strong>Enmo</strong>: the Euclidean Norm Minus One of all accelerometer signals, with negative values rounded to zero.<br>\
            <strong>Anglez</strong>:  a metric derived from individual accelerometer components that refers to the angle of the arm relative to the vertical axis of the body.<br><br></div>',\
                unsafe_allow_html=True)


st.divider() 
# logos kaggle and CMI
img_logos = Image.open(os.path.join("input", "Kaggle_CMI.png"))
col1, col2, col3 = st.columns([1.2, 5, 1.2])
col2.image(img_logos, use_column_width=True)

# links Kaggle, CMI
with col2:
    st.write("Kaggle competition [main page](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states), competition host: [Child Mind Institute](https://childmind.org/)", use_column_width=True)
st.divider()


st.header("Example:mag_right:")
st.markdown("For one series (records of one accelerometer on several days) we plot the features, then the labelled target and our predictions with an XGBoost Classifier.")

st.markdown('Visualization of the features of the series:')
img1 = Image.open(os.path.join("input", "enmo_anglez.png"))
st.image(img1)

st.markdown('Visualization of enmo with sleep periods:zzz: (given by kaggle):')
img2 = Image.open(os.path.join("input", "enmo_target.png"))
st.image(img2)

st.markdown('Visualization of enmo with sleep periods and model predictions:')
img3 = Image.open(os.path.join("input", "enmo_target_prediction.png"))
st.image(img3)



line_break()



st.header("Try it out:rocket:")
file = st.file_uploader("Upload data", type=["csv", "parquet"])
st.markdown("Display a random series from your data without and with predictions. Then you can download the prediction file for all series.")

if file is None:
    st.text("Waiting for upload")
else:
    with st.spinner("Checking file..."):

        # read file
        df = read_file(file.name, file)

        # check df has correct columns and types
        check = check_df(df)
    if check != True:
        st.write('Sorry, the data is not correct:')
        for elem in check:
            st.write(' - ', elem)
        

    else:
    
        with st.spinner("Plotting enmo..."):
            # get id for series we will plot
            id_viz = get_random_id(df)

            # plot enmo
            st.pyplot(plot_enmo(df, id_viz))
        
        with st.spinner("Feature engineering..."):

            # feature engineering
            df_prepared = feature_engineering(df)

        with st.spinner("Loading pipeline..."):

            # load pipeline
            fp_pipeline = os.path.join("input", "pipeline_01.pkl")
            pipeline_1 = pickle.load(open(fp_pipeline, 'rb'))

        with st.spinner("Predictions..."):
            # get predictions
            y_pred, y_probas = get_predictions_probas(df_prepared, pipeline_1)

        with st.spinner ("Building submission dataframe..."):
            df_prediction = build_submission(df_prepared, y_pred, y_probas)

        with st.spinner("Plotting predictions..."):
            st.pyplot(plot_prediction(df_prepared, df_prediction, id_viz))

        
        # display prediction dataset
        st.write("Dataframe of predicted events:")
        st.dataframe(df_prediction.drop(columns=["row_id"]))
        st.markdown("The score column is the confidence of the model in the prediction.")

        with st.spinner("Creating prediction file..."):
            prediction_file = df_prediction.to_csv().encode('utf-8')


        ste.download_button(
            label="Download prediction file",
            data=prediction_file,
            file_name="predictions.csv",
            mime='text/csv'
            )
        


line_break()



st.header("No data? Try sample series:sparkles:")
selected_sample = st.selectbox(
    'Select a sample:',
    ('-', 'Sample 1', 'Sample 2', 'Sample 3'))


dict_choice = {'Sample 1': os.path.join("input", "series_1.csv"),
               'Sample 2': os.path.join("input", "series_2.csv"),
               'Sample 3': os.path.join("input", "series_3.csv")
               }
selected_series = dict_choice.get(selected_sample, None)
if selected_series != None:
    with st.spinner("Loading data..."):

        # read file
        df_2 = pd.read_csv(selected_series)

    with st.spinner("Plotting enmo..."):
        # get id for series we will plot
        id_viz_2 = get_random_id(df_2)

        # plot enmo
        st.pyplot(plot_enmo(df_2, id_viz_2))
    
    with st.spinner("Feature engineering..."):

        # feature engineering
        df_prepared_2 = feature_engineering(df_2)

    with st.spinner("Loading pipeline..."):

        # load pipeline
        fp_pipeline_2 = os.path.join("input", "pipeline_01.pkl")
        pipeline_1 = pickle.load(open(fp_pipeline_2, 'rb'))

    with st.spinner("Predictions..."):
        # get predictions
        y_pred_2, y_probas_2 = get_predictions_probas(df_prepared_2, pipeline_1)

    with st.spinner ("Building submission dataframe..."):
        df_prediction_2 = build_submission(df_prepared_2, y_pred_2, y_probas_2)

    with st.spinner("Plotting predictions..."):
        st.pyplot(plot_prediction(df_prepared_2, df_prediction_2, id_viz_2))

    
    # display prediction dataset
    st.write("Dataframe of predicted events:")
    st.dataframe(df_prediction_2.drop(columns=["row_id"]))
    st.markdown("The score column is the confidence of the model in the prediction.")

    with st.spinner("Creating prediction file..."):
        prediction_file_2 = df_prediction_2.to_csv().encode('utf-8')
        
    ste.download_button(
        label="Download prediction file",
        data=prediction_file_2,
        file_name="predictions.csv",
        mime='text/csv'
        )
line_break()
st.divider()

st.markdown("<div style='text-align: center; color: #505050;'>Sources</div>", unsafe_allow_html=True)
st.write(":gray[The Comprehensive R Archive Network] [Accelerometer data processing with GGIR](https://cran.r-project.org/web/packages/GGIR/vignettes/GGIR.html#4_Inspecting_the_results)")
st.write(":gray[National Library of Medicine] [Segmenting accelerometer data from daily life with unsupervised machine learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6326431/)")
st.write(":gray[Nature - Scientific reports] [Estimating sleep parameters using an accelerometer without sleep diary](https://www.nature.com/articles/s41598-018-31266-z)")
st.divider()

st.markdown("<div style='text-align: center; color: #505050;'>Acknowledgements</div>", unsafe_allow_html=True)
st.markdown(":gray[I would like to thank my colleagues and teachers as well a the Kaggle and Stack Overflow community. They really helped me increase my knowledge and skills.]")
