# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:36:24 2023

@author: Pascal
"""

# =============================================================================
# Import libaries
# =============================================================================

# Standard libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()   
import os
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
import re
from copy import copy
from glob import glob

# skLearn
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import resample

# Vizualization with streamlit
import inspect
import textwrap
import streamlit as st
from PIL import Image


# Define Plot Settings
plt.rcParams.update(
    {
        "axes.facecolor": "white", #"#e6f2ff",
        "axes.spines.right": "True",
        # "axes.labelsize": "15",
        # "axes.titlesize": "20",
        "axes.titleweight": "550",
        "axes.linewidth": "0.5",
        'axes.edgecolor': 'black',
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": "True",
        "ytick.right": "True",
        # 'ytick.major.width': "0.5",
        # 'ytick.major.size': "10",
        # 'xtick.major.width': "0.5",
        # 'xtick.major.size': "10",
        # 'ytick.minor.width': "0.5",
        # 'ytick.minor.size': "5",
        # 'xtick.minor.width': "0.5",
        # 'xtick.minor.size': "5",
        "figure.dpi": "200",
        "figure.frameon": "True",
        "figure.subplot.top": "0.925",
        "savefig.bbox": "tight",
        # "legend.fontsize": "15",
        "legend.frameon": "True",
#         #"legend.shadow": "True",
        "legend.framealpha": "1.0",
        "legend.title_fontsize": "12",
        "figure.facecolor": "white", #"#f2f2f2",
        "figure.titlesize": "12",
        "figure.titleweight": "bold",
        "figure.figsize": "7,5", #"24,20",
        "date.autoformatter.year": "%y",
        "boxplot.notch": "True",
#         'font.serif' : 'Ubuntu',
#         'font.monospace' : 'Ubuntu Mono',
        "font.size": "18",
        "axes.formatter.use_locale": True,
    }
)  # set parameter globally


st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# =============================================================================
# Build vizualization App Header
# =============================================================================
# Streamlit app title
st.title('Explanation Modul: "Enhancing Early Indoor Fire Detection Using Indicative Patterns in Multivariate Time Series Data Based on Multi-Sensor Nodes"')

# =============================================================================
# Decide on Single Node model vs network model
# =============================================================================

# Define options
model_options = ['single_node_approach','network_approach']

# Select box
selected_model = st.selectbox('Select which model approach should be considered', model_options)


# Single node model
if selected_model == 'single_node_approach':
    st.write("Running single node approach...")
    
    
    # =============================================================================
    # File path data
    # =============================================================================

    # Get the current directory
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data')

    file_path = os.path.join(current_dir, 'export/single_node/')

    # =============================================================================
    # Import data
    # =============================================================================

    unique_sensornodes = set()
    unique_scalers = set()

    # Get a list of all CSV files in the folder
    files_shap_fire = glob(os.path.join(file_path, 'df_shap_values_elba_rocket_fire_*'))
    files_shap_nofire = glob(os.path.join(file_path, 'df_shap_values_elba_rocket_nofire_*'))
    files_test = glob(os.path.join(file_path, 'df_test_elba_rocket_*'))

    # Create a dictionary to store your DataFrames
    dfs_shap_fire = {}
    dfs_shap_nofire = {}
    dfs_test = {}


    # Import your DataFrames and store them in the dictionary
    for file in files_shap_fire:
        # Extract sensornode and scaler name from the filename
        filename = os.path.splitext(os.path.basename(file))[0]
        sensornode_name, scaler_name = filename.split('_')[-2:]
        
        # Read the CSV file into a DataFrame and store it in the dictionary
        key = f"{sensornode_name}_{scaler_name}"
        dfs_shap_fire[key] = pd.read_csv(file, index_col=0)
            

    # Import your DataFrames and store them in the dictionary
    for file in files_shap_nofire:
        # Extract sensornode and scaler name from the filename
        filename = os.path.splitext(os.path.basename(file))[0]
        sensornode_name, scaler_name = filename.split('_')[-2:]
        
        # Read the CSV file into a DataFrame and store it in the dictionary
        key = f"{sensornode_name}_{scaler_name}"
        dfs_shap_nofire[key] = pd.read_csv(file, index_col=0)
        
        # Append names to lists
        unique_sensornodes.add(sensornode_name)
        unique_scalers.add(scaler_name)
        

    # Import your DataFrames and store them in the dictionary
    for file in files_test:
        # Extract sensornode and scaler name from the filename
        filename = os.path.splitext(os.path.basename(file))[0]
        sensornode_name, scaler_name = filename.split('_')[-2:]
        
        # Read the CSV file into a DataFrame and store it in the dictionary
        key = f"{sensornode_name}_{scaler_name}"
        dfs_test[key] = pd.read_csv(file, index_col=0)


    # Convert set to list
    list_sensornodes = sorted(list(unique_sensornodes))
    list_scalers = list(unique_scalers)

    # =============================================================================
    # Data Selection
    # =============================================================================
    # Define Columns
    col1, col2 = st.columns([1, 2])

    with col1: 
        # Dropdown widget to select the sensornode for plotting data
        sensornode = st.selectbox('Select Sensor Node', list_sensornodes)
        scaler = st.selectbox('Select scaler for model', list_scalers)
        
        key = f"{sensornode}_{scaler}"
        
        df_test = dfs_test.get(key)
        
        # Dropdown widget to select ground truth label
        ground_truth_label = st.selectbox('Select Ground Truth Label', df_test.fire_label.unique().tolist())

    with col2:
        # Dropdown widget to select model label
        model_prediction_label = st.selectbox('Select Model Prediction Label', df_test.model_prediction.unique().tolist())
        
        # Group data based on Label selection
        df_test_loc_label = df_test.copy()
        df_test_loc_label = df_test_loc_label.loc[
            (df_test_loc_label.fire_label == ground_truth_label) & (df_test_loc_label.model_prediction == model_prediction_label)
             ]
        
        # Dropdown widget to select the interval
        selected_interval = st.selectbox('Select Interval', df_test_loc_label.index.get_level_values('interval_label').unique().tolist())
        
        # Anzeigebox für Predicted label des models für ausgewähltes Interval
        model_prediction = df_test.loc[df_test.index == selected_interval]['model_prediction'].unique()[0]
        #st.info(f"Predicted Label from Model: {model_prediction}")
            
    # =============================================================================
    # Plot Picture of Sensor node positions
    # ============================================================================= 

    # List all files in the folder
    figures_file_path = os.path.join(current_dir, 'data/figures/')

    # List all files in the folder
    files = os.listdir(figures_file_path)

    # Filter files based on the sensor node number
    matching_files = [file for file in files if file.endswith(f"_{sensornode}.png")]

    # Check if there is a matching file
    if matching_files:
        # Plot the matching file
        matching_file = matching_files[0]  # Assuming there's only one matching file
        file_path = os.path.join(figures_file_path, matching_file)

        # Read and plot the image
        img = Image.open(file_path)
        st.image(img, caption='Sensor Node Positions',use_column_width=True)

    # =============================================================================
    # Plot time series sensor measurmenents
    # =============================================================================
            
    # Define data for Sensor measurements plot
    data_temp_measurements = df_test.copy()
    data_temp_measurements = data_temp_measurements.loc[data_temp_measurements.index == selected_interval]

    # Display time series plot
    st.subheader(f"Time Series Plot of Sensor Measurements for Sensor Node {sensornode}")

    # RAW plots
    fig = plt.figure(figsize=(6, 4))

    # figure           
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=False) #, ax5
    # title
    ax1.set_title('Sensornode: ' + str(sensornode) + ', Interval: ' + str(selected_interval))
    # set axes plots
    ax1.plot(data_temp_measurements.timepoints, data_temp_measurements['CO_Room'], '-', label='CO',color='blue')
    ax2.plot(data_temp_measurements.timepoints, data_temp_measurements['H2_Room'], '-', label='H2',color='red')
    ax3.plot(data_temp_measurements.timepoints, data_temp_measurements['VOC_Room_RAW'], '-', label='VOC',color='purple')
    ax4.plot(data_temp_measurements.timepoints, data_temp_measurements['PM05_Room'],'-', label='PM05',color='black')
    ax5.plot(data_temp_measurements.timepoints, data_temp_measurements['PM10_Room'], '-', label='PM10',color='black')
    # legend
    f.legend(bbox_to_anchor=(0.9, 0.6), loc='upper left', frameon=True, title = 'feature')
    # Adding y labels
    ax1.set_ylabel(r'$ppm$',rotation=0, labelpad=30, fontsize=10)
    ax2.set_ylabel(r'$ppm$',rotation=0, labelpad=30, fontsize=10)
    ax3.set_ylabel('$A.U.$',rotation=0, labelpad=30, fontsize=10)
    ax4.set_ylabel(r'$cm^{-3}$',rotation=0,labelpad=30, fontsize=10)
    ax5.set_ylabel(r'$cm^{-3}$',rotation=0,labelpad=30, fontsize=10)
    ax5.set_xlabel('timepoints',rotation=0,labelpad=10, fontsize=10)

    # Show plot
    plt.show()

    # Display the subplots in Streamlit
    st.pyplot()

    # =============================================================================
    # Define data for explanation
    # =============================================================================
    # Define data for SHAP plot

    if model_prediction == 'Fire':
        data_temp_shap_values = dfs_shap_fire.get(key).copy()
        data_temp_shap_values = data_temp_shap_values.loc[data_temp_shap_values.index == selected_interval]
        
    else:
        if model_prediction == 'NoFire':
            data_temp_shap_values = dfs_shap_nofire.get(key).copy()
            data_temp_shap_values = data_temp_shap_values.loc[data_temp_shap_values.index == selected_interval]
        

    # Set the index to 'Timepoints' for better visualization
    data_temp_shap_values.set_index('timepoints', inplace=True)

    # =============================================================================
    # Plot Shap values as sum of all sensornodes
    # =============================================================================
    #define the colormap with clipping values
    my_cmap = copy(plt.cm.OrRd)
    # my_cmap.set_over("white")
    my_cmap.set_under("white")

    # Pivot the DataFrame for heatmap plotting
    heatmap_data = data_temp_shap_values.T

    # Display time series plot
    st.subheader("SHAP Explanation")

    # Create the heatmap
    plt.figure(figsize=(6, 4))
    heatmap = sns.heatmap(heatmap_data, annot=False, cmap=my_cmap, vmin= -0.2, vmax=0.5, cbar=True)

    # Rotate y-axis labels to be horizontal
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

    plt.xlabel('Timepoints')
    plt.ylabel('Feature')
    plt.show()

    # Display the subplots in Streamlit
    st.pyplot()

    # =============================================================================
    # Confuion Matrix
    # =============================================================================

    # Extract the first label in each interval for both ground truth and predicted label
    ground_truth_labels = list(df_test.groupby('interval_label')['fire_label'].first())
    predicted_labels = list(df_test.groupby('interval_label')['model_prediction'].first())

    # Define your positive and negative class labels
    positive_class = "Fire"
    negative_class = "NoFire"

    # Create confusion matrix with specified labels
    cm = confusion_matrix(ground_truth_labels, predicted_labels, labels=[negative_class, positive_class]) # Order important: negative class, positive Class

    st.subheader("Model Performance")

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[negative_class, positive_class], yticklabels=[negative_class, positive_class])
    plt.xlabel('Predicted')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Display other relevant metrics if needed (e.g., precision, recall, accuracy)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'Accuracy: {accuracy:.2f}')
    
    
    
# Network model
elif selected_model == 'network_approach':
    st.write("Running network approach...")

    # =============================================================================
    # File path data
    # =============================================================================
    
    # Get the current directory
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data')

    file_path = os.path.join(current_dir, 'export/network/')

    # =============================================================================
    # Import data
    # =============================================================================
    
    # Get a list of all CSV files in the folder
    files_shap_fire = glob(os.path.join(file_path, 'df_shap_values_elba_rocket_fire_network_*'))
    files_shap_nofire = glob(os.path.join(file_path, 'df_shap_values_elba_rocket_nofire_network_*'))
    files_test = glob(os.path.join(file_path, 'df_test_elba_rocket_network_*'))

    # Create a dictionary to store your DataFrames
    dfs_shap_fire = {}
    dfs_shap_nofire = {}
    dfs_test = {}

    # Initialize set
    unique_scalers = set()
    
    
    # Import your DataFrames and store them in the dictionary
    for file in files_shap_fire:
        # Extract sensornode and scaler name from the filename
        filename = os.path.splitext(os.path.basename(file))[0]
        
        parts = filename.split('_')
        scaler_name = parts[-1]  # Get the last part as a string
        #scaler_name = filename.split('_')[-1:]
        
        # Read the CSV file into a DataFrame and store it in the dictionary
        key = f"{scaler_name}"
        dfs_shap_fire[key] = pd.read_csv(file, index_col=0)
            

    # Import your DataFrames and store them in the dictionary
    for file in files_shap_nofire:
    
        # Extract sensornode and scaler name from the filename
        filename = os.path.splitext(os.path.basename(file))[0]
        
        parts = filename.split('_')
        scaler_name = parts[-1]  # Get the last part as a string
        #scaler_name = filename.split('_')[-1:]
        
        # Read the CSV file into a DataFrame and store it in the dictionary
        key = f"{scaler_name}"
        dfs_shap_nofire[key] = pd.read_csv(file, index_col=0)
        
        # Append scaler names to lists
        unique_scalers.add(scaler_name)
        

    # Import your DataFrames and store them in the dictionary
    for file in files_test:
        # Extract sensornode and scaler name from the filename
        filename = os.path.splitext(os.path.basename(file))[0]
                
        parts = filename.split('_')
        scaler_name = parts[-1]  # Get the last part as a string
        #scaler_name = filename.split('_')[-1:]
        
        # Read the CSV file into a DataFrame and store it in the dictionary
        key = f"{scaler_name}"
        dfs_test[key] = pd.read_csv(file, index_col=0)
        
        
    # Convert set to list
    list_scalers = list(unique_scalers)

    # =============================================================================
    # Build vizualization App Header
    # =============================================================================

    # Dropdown widget to select the sensornode for plotting data
    list_sensornodes = ['8','9','10','11','12','13','14','15','16'] 

    # Define Columns
    col1, col2 = st.columns([1, 2])

    with col1: 
    
        # Dropdown widget to select the sensornode for plotting data
        scaler = st.selectbox('Select scaler for model', list_scalers)
        
        key = f"{scaler}"
        
        df_test = dfs_test.get(key)
        
        # Dropdown widget to select ground truth label
        ground_truth_label = st.selectbox('Select Ground Truth Label', df_test.fire_label.unique().tolist())
        

    with col2:     
    
        # Dropdown widget to select model label
        model_prediction_label = st.selectbox('Select Model Prediction Label', df_test.model_prediction.unique().tolist())
        
        # Group data based on Label selection
        df_test_loc_label = df_test.copy()
        df_test_loc_label = df_test_loc_label.loc[
            (df_test_loc_label.fire_label == ground_truth_label) & (df_test_loc_label.model_prediction == model_prediction_label)
             ]
        
        # Dropdown widget to select the interval
        selected_interval = st.selectbox('Select an Interval', df_test_loc_label.index.get_level_values('interval_label').unique().tolist())
        
        # Anzeigebox für Predicted label des models für ausgewähltes Interval
        model_prediction = df_test.loc[df_test.index == selected_interval]['model_prediction'].unique()[0]
        #st.info(f"Predicted Label from Model: {model_prediction}")

    # =============================================================================
    # Plot Picture of Sensor node positions
    # =============================================================================

    # Define the path to the image
    figures_file_path = os.path.join(current_dir, 'data/figures/')
    overview_filename = 'sensor_node_positions_ELBA_overview.png'
    overview_filepath = os.path.join(figures_file_path, overview_filename)

    # Check if the file exists and display it
    if os.path.exists(overview_filepath):
        img = Image.open(overview_filepath)
        st.image(img, caption='Sensor Node Positions Overview', use_column_width=True)
    else:
        st.warning(f"File {overview_filename} not found in {figures_file_path}.")

    # # =============================================================================
    # # Plot time series sensor measurmenents
    # # =============================================================================
            
    # # Define data for Sensor measurements plot
    # data_temp_measurements = df_test.copy()
    # data_temp_measurements = data_temp_measurements.loc[data_temp_measurements.index == selected_interval]

    # # Display time series plot
    # st.subheader(f"Time Series Plot of Sensor Measurements for Sensor Node {sensornode}")

    # # RAW plots
    # fig = plt.figure()

    # # figure           
    # f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=False)
    # # title
    # ax1.set_title('Sensornode: ' + str(sensornode) + ', Interval' + str(selected_interval))
    # # set axes plots
    # ax1.plot(data_temp_measurements.timepoints, data_temp_measurements[f'CO_Room_{sensornode}'], '-', label='CO',color='blue')
    # ax2.plot(data_temp_measurements.timepoints, data_temp_measurements[f'H2_Room_{sensornode}'], '-', label='H2',color='red')
    # ax3.plot(data_temp_measurements.timepoints, data_temp_measurements[f'VOC_Room_RAW_{sensornode}'], '-', label='VOC',color='purple')
    # ax4.plot(data_temp_measurements.timepoints, data_temp_measurements[f'PM05_Room_{sensornode}'],'-', label='PM05',color='black')
    # ax5.plot(data_temp_measurements.timepoints, data_temp_measurements[f'PM10_Room_{sensornode}'], '-', label='PM10',color='black')
    # # legend
    # f.legend(bbox_to_anchor=(0.95, 0.7), loc='upper left', frameon=True, title = 'Measurant')
    # # Adding y labels
    # ax1.set_ylabel(r'$ppm$',rotation=0, labelpad=30)
    # ax2.set_ylabel(r'$ppm$',rotation=0, labelpad=30)
    # ax3.set_ylabel('$A.U.$',rotation=0, labelpad=30)
    # ax4.set_ylabel(r'$cm^{-3}$',rotation=0,labelpad=30)
    # ax5.set_ylabel(r'$cm^{-3}$',rotation=0,labelpad=30)
    # ax5.set_xlabel('timepoints',rotation=0,labelpad=10)

    # # # Define lower and upper limits for fixed y-axis
    # # # CO
    # # lower_limit_CO = 0
    # # upper_limit1_CO = 10
    # # # H2
    # # lower_limit_H2 = 0
    # # upper_limit1_H2 = 10
    # # # VOC_RAW
    # # lower_limit_VOC_RAW = 0
    # # upper_limit1_VOC_RAW = 15
    # # # PM05
    # # lower_limit_PM05 = 0
    # # upper_limit1_PM05 = 30000
    # # # PM10
    # # lower_limit_PM10 = 0
    # # upper_limit1_PM10 = 35000


    # # # Set y-limits for each axis
    # # ax1.set_ylim([lower_limit_CO, upper_limit1_CO])
    # # ax2.set_ylim([lower_limit_H2, upper_limit1_H2])
    # # ax3.set_ylim([lower_limit_VOC_RAW, upper_limit1_VOC_RAW])
    # # ax4.set_ylim([lower_limit_PM05, upper_limit1_PM05])
    # # ax5.set_ylim([lower_limit_PM10, upper_limit1_PM10])

    # # plt.grid(True)+

    # # Show plot
    # plt.show()

    # # Display the subplots in Streamlit
    # st.pyplot()

    # =============================================================================
    # Define data for explanation
    # =============================================================================
    
    # Define data for SHAP plot

    if model_prediction == 'Fire':
        data_temp_shap_values = dfs_shap_fire.get(key).copy()
        data_temp_shap_values = data_temp_shap_values.loc[data_temp_shap_values.index == selected_interval]
        
    else:
        if model_prediction == 'NoFire':
            data_temp_shap_values = dfs_shap_nofire.get(key).copy()
            data_temp_shap_values = data_temp_shap_values.loc[data_temp_shap_values.index == selected_interval]
        

    # Set the index to 'Timepoints' for better visualization
    data_temp_shap_values.set_index('timepoints', inplace=True)

    # =============================================================================
    # Plot Shap values as sum of all sensornodes
    # =============================================================================

    #define the colormap with clipping values
    my_cmap = copy(plt.cm.OrRd)
    # my_cmap.set_over("white")
    my_cmap.set_under("white")

    # Initialize lists to store the data
    timepoints = []
    sensor_ids = []
    measurements = []
    shap_values = []

    # Iterate over the columns and create lists for each attribute
    for col in data_temp_shap_values.columns:
        if col != 'timepoints':
            sensor_id = col.split('_')[-1]  # Extract the Sensor_ID
            measurement = col.rsplit('_', 1)[0]  # Extract all strings before the last "_"
            timepoints.extend(data_temp_shap_values.index)
            sensor_ids.extend([sensor_id] * len(data_temp_shap_values))
            measurements.extend([measurement] * len(data_temp_shap_values))
            shap_values.extend(data_temp_shap_values[col].tolist())

    # Create the transformed DataFrame
    transformed_df = pd.DataFrame({
        'timepoints': timepoints,
        'Sensor_ID': sensor_ids,
        'measurement': measurements,
        'shap_value': shap_values
    })


    # Filter the DataFrame to exclude negative values when calculating the sum
    # transformed_df = transformed_df[transformed_df['shap_value'] >= 0]

    # Pivot the DataFrame to calculate the sum of 'shap_value' for each 'measurement' and 'timepoints'
    grouped_df = transformed_df.pivot_table(index=['measurement', 'timepoints'], values='shap_value', aggfunc='sum').reset_index()


    # Pivot the DataFrame for heatmap plotting
    heatmap_data = grouped_df.pivot(index='measurement', columns='timepoints', values='shap_value')


    # Display time series plot
    st.subheader("Network explanation")

    # Create the heatmap
    plt.figure(figsize=(6, 4))
    heatmap = sns.heatmap(heatmap_data, annot=False, cmap=my_cmap, vmin= -0.2, vmax=0.5, cbar=True)

    # Rotate y-axis labels to be horizontal
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

    plt.xlabel('Timepoints')
    plt.ylabel('Feature')
    plt.show()

    # Display the subplots in Streamlit
    st.pyplot()   
    

    # # =============================================================================
    # # Explanation for selected Sensornode
    # # =============================================================================

    # # Pivot the DataFrame for heatmap plotting
    # heatmap_data = data_temp_shap_values.filter(regex=f'_{sensornode}$', axis=1).T

    # # Display time series plot
    # st.subheader(f"Explanation Sensornode {sensornode}")

    # # Create the heatmap
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(heatmap_data, annot=False, cmap=my_cmap, vmin= -0.2, vmax=0.5, cbar=True)
    # plt.xlabel('Timepoints')
    # plt.ylabel('Measurement')
    # plt.show()

    # # Display the subplots in Streamlit
    # st.pyplot()

    # # =============================================================================
    # # # Plot Headmap Shap values sensornode- wise
    # # =============================================================================

    # # Get a list of unique sensors
    # sensors = list_sensornodes

    # # Create subplots in a 3x3 matrix
    # nrows = 3
    # ncols = 3
    # fig, axes = plt.subplots(nrows, ncols, figsize=(20, 14))

    # # Flatten the axes array for easy iteration
    # axes = axes.ravel()

    # # Create a dictionary to store legend handles and labels
    # legend_handles_labels = {}

    # # Flag to track if we're in the first row of subplots
    # first_row = True

    # # Iterate through sensors and create heatmaps
    # for i, sensor in enumerate(sensors):
        # # Filter columns for the current sensor
        # columns_for_sensor = [col for col in data_temp_shap_values.columns if f"_{sensor}" in col]

        # # Transpose the filtered DataFrame
        # df_transposed = data_temp_shap_values[columns_for_sensor].T


        # # Create a heatmap for the current sensor, storing the handle for the legend
        # heatmap = sns.heatmap(df_transposed, annot=False, cmap=my_cmap, vmin= -0.2, vmax=0.5, cbar=True, ax=axes[i])
        

        # # Customize labels and title for each subplot
        # y_labels = df_transposed.index  # Get the Y-axis labels
        
        # # Modify Y-axis labels to remove characters after the last "_"
        # y_labels = [label.rsplit('_', 1)[0] for label in y_labels]
        
        # # Remove "_Room" where it is included in a label
        # y_labels = [label.replace('_Room', '') for label in y_labels]
        
        # if first_row:
            # axes[i].set_ylabel('Features')
            # first_row = False
        # else:
            # # Suppress Y-axis labels and annotations for subplots not in the first row
            # axes[i].set_yticklabels([])
        
        # # Set the modified Y-axis labels
        # axes[i].set_yticklabels(y_labels)
        
        # axes[i].set_xlabel('Timepoints')
        # axes[i].set_title(f'Sensor Node {sensor}')
        
        # # If we've reached the end of the row, reset the first_row flag
        # if (i + 1) % ncols == 0:
            # first_row = True

    # # Remove any unused subplots if the number of sensors is less than 9
    # for i in range(len(sensors), nrows * ncols):
        # fig.delaxes(axes[i])

    # # Adjust spacing between subplots
    # plt.tight_layout()

    # # Show plot
    # plt.show()

    # # Display the subplots in Streamlit
    # st.pyplot()


    # =============================================================================
    # Confuion Matrix
    # =============================================================================

    # Extract the first label in each interval for both ground truth and predicted label
    ground_truth_labels = list(df_test.groupby('interval_label')['fire_label'].first())
    predicted_labels = list(df_test.groupby('interval_label')['model_prediction'].first())


    # Define your positive and negative class labels
    positive_class = "Fire"
    negative_class = "NoFire"

    # Create confusion matrix with specified labels
    cm = confusion_matrix(ground_truth_labels, predicted_labels, labels=[negative_class, positive_class]) # Order important: negative class, positive Class

    st.subheader("Model Performance")

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[negative_class, positive_class], yticklabels=[negative_class, positive_class])
    plt.xlabel('Predicted')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Display other relevant metrics if needed (e.g., precision, recall, accuracy)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'Accuracy: {accuracy:.2f}')






