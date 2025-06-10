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
import configparser
import os
from pathlib import Path
import argparse
import logging
import datetime
import inspect
from sklearn.utils import resample
import re
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objs as go
import math
from glob import glob

# skLearn
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Sktime
from sktime.classification.kernel_based import RocketClassifier
from sktime.datatypes import check_raise
from sktime.datatypes import mtype
from sktime.datatypes import check_is_mtype
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.transformations.series.summarize import SummaryTransformer
from sktime.datatypes import convert_to
from sktime.datatypes import convert
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# Additional
import matplotlib.dates as mdates
import joblib
import time # to claculate the runtime of models
from pathlib import Path 
import pymannkendall as mk # Kendall tau trend package

# Internal Packages
from analyse_df import analyse_df
from rename_columns import rename_columns
import plot_settings

# SHAP Explanation
import shap

# Vizualization with streamlit
import inspect
import textwrap
import streamlit as st
from PIL import Image

# Standard libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()   
import configparser
import os
from pathlib import Path
import argparse
import logging
import datetime
import inspect
from sklearn.utils import resample
import re
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objs as go
import math
from copy import copy

# Plot settings
import scienceplots

#plt.style.use("seaborn-ticks")
plt.style.use(['science']) #,'no-latex'
# plt.rc("text", usetex=True)
# plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Noto Sans"]})
# plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage[sfdefault]{noto}")

plt.rcParams.update(
    {
        "axes.facecolor": "white", #"#e6f2ff",
        "axes.spines.right": "True",
        "axes.labelsize": "15",
        "axes.titlesize": "20",
        "axes.titleweight": "550",
        "axes.linewidth": "0.5",
        'axes.edgecolor': 'black',
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": "True",
        "ytick.right": "True",
        'ytick.major.width': "0.5",
        'ytick.major.size': "10",
        'xtick.major.width': "0.5",
        'xtick.major.size': "10",
        'ytick.minor.width': "0.5",
        'ytick.minor.size': "5",
        'xtick.minor.width': "0.5",
        'xtick.minor.size': "5",
        "figure.dpi": "300",
        "figure.frameon": "True",
        "figure.subplot.top": "0.925",
        "savefig.bbox": "tight",
        "legend.fontsize": "15",
        "legend.frameon": "True",
#         #"legend.shadow": "True",
        "legend.framealpha": "1.0",
        "legend.title_fontsize": "18",
        "figure.facecolor": "white", #"#f2f2f2",
        "figure.titlesize": "24",
        "figure.titleweight": "bold",
        "figure.figsize": "14,10", #"24,20",
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

#define the colormap with clipping values
my_cmap = copy(plt.cm.OrRd)
my_cmap.set_over("white")
my_cmap.set_under("white")

# =============================================================================
# File path data
# =============================================================================

# Get the current directory
current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'export/network/')

# =============================================================================
# Import data
# =============================================================================

# Import SHAP values
df_shap_values_fire = pd.read_csv(os.path.join(data_path, 'df_shap_values_elba_rocket_fire_network_NoScaling.csv'), index_col=0)
df_shap_values_nofire = pd.read_csv(os.path.join(data_path, 'df_shap_values_elba_rocket_nofire_network_NoScaling.csv'), index_col=0)

# import test data
df_test = pd.read_csv(os.path.join(data_path, 'df_test_elba_rocket_network_NoScaling.csv'), index_col=0)

# =============================================================================
# Build vizualization App Header
# =============================================================================

# Dropdown widget to select the sensornode for plotting data
list_sensornodes = ['8','9','10','11','12','13','14','15','16']

# Streamlit app title
st.title('Explanation Modul: "Detecting Early Fire Indicator Patterns in Multivariate Time Series" ')

# Define Columns
col1, col2 = st.columns([1, 2])

with col1: 
    # Dropdown widget to select ground truth label
    ground_truth_label = st.selectbox('Select Ground Truth Label', df_test.fire_label.unique().tolist())
    
    # Dropdown widget to select model label
    model_prediction_label = st.selectbox('Select Model Prediction Label', df_test.model_prediction.unique().tolist())
    
    # Group data based on Label selection
    df_test_loc_label = df_test.copy()
    df_test_loc_label = df_test_loc_label.loc[
        (df_test_loc_label.fire_label == ground_truth_label) & (df_test_loc_label.model_prediction == model_prediction_label)
         ]

with col2:     
    # Select sensor node
    sensornode = st.selectbox('Select Sensor Node', list_sensornodes)
    
    # Dropdown widget to select the interval
    selected_interval = st.selectbox('Select an Interval', df_test_loc_label.index.get_level_values('interval_label').unique().tolist())
    
    # Anzeigebox für Predicted label des models für ausgewähltes Interval
    model_prediction = df_test.loc[df_test.index == selected_interval]['model_prediction'].unique()[0]
    st.info(f"Predicted Label from Model: {model_prediction}")

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
fig = plt.figure()

# figure           
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=False)
# title
ax1.set_title('Sensornode: ' + str(sensornode) + ', Interval' + str(selected_interval))
# set axes plots
ax1.plot(data_temp_measurements.timepoints, data_temp_measurements[f'CO_Room_{sensornode}'], '-', label='CO',color='blue')
ax2.plot(data_temp_measurements.timepoints, data_temp_measurements[f'H2_Room_{sensornode}'], '-', label='H2',color='red')
ax3.plot(data_temp_measurements.timepoints, data_temp_measurements[f'VOC_Room_RAW_{sensornode}'], '-', label='VOC',color='purple')
ax4.plot(data_temp_measurements.timepoints, data_temp_measurements[f'PM05_Room_{sensornode}'],'-', label='PM05',color='black')
ax5.plot(data_temp_measurements.timepoints, data_temp_measurements[f'PM10_Room_{sensornode}'], '-', label='PM10',color='black')
# legend
f.legend(bbox_to_anchor=(0.95, 0.7), loc='upper left', frameon=True, title = 'Measurant')
# Adding y labels
ax1.set_ylabel(r'$ppm$',rotation=0, labelpad=30)
ax2.set_ylabel(r'$ppm$',rotation=0, labelpad=30)
ax3.set_ylabel('$A.U.$',rotation=0, labelpad=30)
ax4.set_ylabel(r'$cm^{-3}$',rotation=0,labelpad=30)
ax5.set_ylabel(r'$cm^{-3}$',rotation=0,labelpad=30)
ax5.set_xlabel('timepoints',rotation=0,labelpad=10)

# # Define lower and upper limits for fixed y-axis
# # CO
# lower_limit_CO = 0
# upper_limit1_CO = 10
# # H2
# lower_limit_H2 = 0
# upper_limit1_H2 = 10
# # VOC_RAW
# lower_limit_VOC_RAW = 0
# upper_limit1_VOC_RAW = 15
# # PM05
# lower_limit_PM05 = 0
# upper_limit1_PM05 = 30000
# # PM10
# lower_limit_PM10 = 0
# upper_limit1_PM10 = 35000


# # Set y-limits for each axis
# ax1.set_ylim([lower_limit_CO, upper_limit1_CO])
# ax2.set_ylim([lower_limit_H2, upper_limit1_H2])
# ax3.set_ylim([lower_limit_VOC_RAW, upper_limit1_VOC_RAW])
# ax4.set_ylim([lower_limit_PM05, upper_limit1_PM05])
# ax5.set_ylim([lower_limit_PM10, upper_limit1_PM10])

# plt.grid(True)+

# Show plot
plt.show()

# Display the subplots in Streamlit
st.pyplot()

# =============================================================================
# Define data for explanation
# =============================================================================
# Define data for SHAP plot
if model_prediction == 'Fire':
    data_temp_shap_values = df_shap_values_fire.copy()
    data_temp_shap_values = data_temp_shap_values.loc[data_temp_shap_values.index == selected_interval]
    
else:
    if model_prediction == 'NoFire':
        data_temp_shap_values = df_shap_values_nofire.copy()
        data_temp_shap_values = data_temp_shap_values.loc[data_temp_shap_values.index == selected_interval]
    

# Set the index to 'Timepoints' for better visualization
data_temp_shap_values.set_index('timepoints', inplace=True)

# =============================================================================
# Plot Shap values as sum of all sensornodes
# =============================================================================

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
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=False, cmap=my_cmap, vmin= -0.2, vmax=0.5, cbar=True)
plt.xlabel('Timepoints')
plt.ylabel('Measurement')
plt.show()

# Display the subplots in Streamlit
st.pyplot()

# =============================================================================
# Explanation for selected Sensornode
# =============================================================================

# Pivot the DataFrame for heatmap plotting
heatmap_data = data_temp_shap_values.filter(regex=f'_{sensornode}$', axis=1).T

# Display time series plot
st.subheader(f"Explanation Sensornode {sensornode}")

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=False, cmap=my_cmap, vmin= -0.2, vmax=0.5, cbar=True)
plt.xlabel('Timepoints')
plt.ylabel('Measurement')
plt.show()

# Display the subplots in Streamlit
st.pyplot()

# =============================================================================
# # Plot Headmap Shap values sensornode- wise
# =============================================================================

# Get a list of unique sensors
sensors = list_sensornodes

# Create subplots in a 3x3 matrix
nrows = 3
ncols = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 14))

# Flatten the axes array for easy iteration
axes = axes.ravel()

# Create a dictionary to store legend handles and labels
legend_handles_labels = {}

# Flag to track if we're in the first row of subplots
first_row = True

# Iterate through sensors and create heatmaps
for i, sensor in enumerate(sensors):
    # Filter columns for the current sensor
    columns_for_sensor = [col for col in data_temp_shap_values.columns if f"_{sensor}" in col]

    # Transpose the filtered DataFrame
    df_transposed = data_temp_shap_values[columns_for_sensor].T


    # Create a heatmap for the current sensor, storing the handle for the legend
    heatmap = sns.heatmap(df_transposed, annot=False, cmap=my_cmap, vmin= -0.2, vmax=0.5, cbar=True, ax=axes[i])
    

    # Customize labels and title for each subplot
    y_labels = df_transposed.index  # Get the Y-axis labels
    
    # Modify Y-axis labels to remove characters after the last "_"
    y_labels = [label.rsplit('_', 1)[0] for label in y_labels]
    
    # Remove "_Room" where it is included in a label
    y_labels = [label.replace('_Room', '') for label in y_labels]
    
    if first_row:
        axes[i].set_ylabel('Features')
        first_row = False
    else:
        # Suppress Y-axis labels and annotations for subplots not in the first row
        axes[i].set_yticklabels([])
    
    # Set the modified Y-axis labels
    axes[i].set_yticklabels(y_labels)
    
    axes[i].set_xlabel('Timepoints')
    axes[i].set_title(f'Sensor Node {sensor}')
    
    # If we've reached the end of the row, reset the first_row flag
    if (i + 1) % ncols == 0:
        first_row = True

# Remove any unused subplots if the number of sensors is less than 9
for i in range(len(sensors), nrows * ncols):
    fig.delaxes(axes[i])

# Adjust spacing between subplots
plt.tight_layout()

# Show plot
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
plt.figure(figsize=(8, 6))
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

