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

# =============================================================================
# File path data
# =============================================================================

# Get the current directory
current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'data')

export_path = os.path.join(current_dir, 'export')

# =============================================================================
# Import data
# =============================================================================

# Import labeled data set
file_name = 'elba_dataset_pp_4.csv'
data_file_path = os.path.join(data_path, file_name)
df = pd.read_csv(data_file_path, index_col=0)


# =============================================================================
# Build vizualization App Header
# =============================================================================

# Streamlit app title
st.title('Module for Analysing Ground Truth Labels')

# Define Columns
col1, col2 = st.columns([1, 2])

with col1: 
    
    # Dropdown widget to select the sensornode for plotting data
    list_sensornodes = df.Sensor_ID.unique().tolist()
    sensornode = st.selectbox('Select Sensor Node:', list_sensornodes)
    
    # Group data based on Sensor Node selection
    df_loc_sensor = df.copy()
    df_loc_sensor = df_loc_sensor.loc[df_loc_sensor.Sensor_ID == sensornode]
    
    # Dropdown widget to select the label
    selected_label = st.selectbox('Select Ground Truth Label for Filtering', df.fire_label.unique().tolist())
    
    # Group data based on Label selection
    df_loc_sensor_label = df_loc_sensor.copy()
    df_loc_sensor_label = df_loc_sensor_label.loc[df_loc_sensor_label.fire_label == selected_label]
    
    # Dropdown widget to select the interval
    selected_interval = st.selectbox('Select an Interval', df_loc_sensor_label.Interval_label.unique().tolist())   
 
# =============================================================================
# Plot Picture of Sensor node positions
# =============================================================================
with col2: 
    # Display time series plot
    image = Image.open('./data/SK_GTE_Positionen_ELBA.PNG')
    st.image(image, caption='Sensor Node Postiions',use_column_width=True)

# =============================================================================
# Plot time series sensor measurmenents
# =============================================================================
        
# Define data for Sensor measurements plot
data_temp_measurements = df_loc_sensor.copy()
data_temp_measurements = data_temp_measurements.loc[data_temp_measurements.Interval_label == selected_interval]

# Display time series plot
st.subheader(f"Time Series Plot of Sensor Measurements for Sensor Node {sensornode}")

# RAW plots
fig = plt.figure()

# figure           
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=False)
# title
ax1.set_title('Sensornode: ' + str(sensornode) + ', Interval' + str(selected_interval))
# set axes plots
ax1.plot(data_temp_measurements.Date, data_temp_measurements['CO_Room'], '-', label='CO',color='blue')
ax2.plot(data_temp_measurements.Date, data_temp_measurements['H2_Room'], '-', label='H2',color='red')
ax3.plot(data_temp_measurements.Date, data_temp_measurements['VOC_Room_RAW'], '-', label='VOC',color='purple')
ax4.plot(data_temp_measurements.Date, data_temp_measurements['PM05_Room'],'-', label='PM05',color='black')
ax5.plot(data_temp_measurements.Date, data_temp_measurements['PM10_Room'], '-', label='PM10',color='black')
# legend
f.legend(bbox_to_anchor=(0.95, 0.7), loc='upper left', frameon=True, title = 'Measurant')
# Adding y labels
ax1.set_ylabel(r'$ppm$',rotation=0, labelpad=30)
ax2.set_ylabel(r'$ppm$',rotation=0, labelpad=30)
ax3.set_ylabel('$A.U.$',rotation=0, labelpad=30)
ax4.set_ylabel(r'$cm^{-3}$',rotation=0,labelpad=30)
ax5.set_ylabel(r'$cm^{-3}$',rotation=0,labelpad=30)
ax5.set_xlabel('timepoints',rotation=0,labelpad=10)

# Show plot
plt.show()

# Display the subplots in Streamlit
st.pyplot()


