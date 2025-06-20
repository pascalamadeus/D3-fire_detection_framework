# D3-fire_detection_framework
 This repository contains the code used in the paper "Enhancing Early Indoor Fire Detection Using Indicative Patterns in Multivariate Time Series Data Based on Multi-Sensor Nodes"
 
 ## Overview
 
 
 
 
 ## Abstract
 
 Multi-sensor technology is becoming increasingly accessible, resulting in more complex data structures that require reasonable analysis in order to extract crucial information. In this study, we propose a novel approach for enhancing early indoor fire detection using multi-sensor nodes. We model the task of early indoor fire detection as a binary classification problem in multivariate time series (MTS) data to capture early fire indicator patterns in their (a) dimension of emissions, (b) dimension of sensor placement and (c) dimension of time. We combine a prediction component based on ROCKET (RandOm Convolutional KErnel Transformation) and an explanation component based on SHAP (SHapley Additive exPlanations) as a novel approach to provide robust early fire detection and model agnostic explanations simultaneously in one system. Based on five different sensor measurements (CO, H2, VOC (volatile organic compounds), PM05, and PM10 (particulate matter)) mea-sured at nine different sensor node positions, our results highlight VOC and PM as the main early fire indicators, outperforming gases like H2 or CO. We demonstrate that the dimension of time is just as important to early indoor fire detection as the absolute sensor measurement value (dimension of emissions). Our findings indicate that a single node approach is more suitable than a network approach for the purpose of early indoor fire detec-tion. Furthermore, we could confirm the presence of numerical underflow in the KernalSHAP explanation for MTSC, which significantly reduces SHAP values (by up to 61 \%) at longer interval lengths. This effect also depends on the total number of input features.
 
 ## Setup
 
 Setup a python env based on python 3.12.3. The following package versions need to be installed in that env:
 
- shap 0.42.1
- sktime 0.33.0
- xgboost 1.7.3
- streamlit 1.24.1
 
 ## Usage
 
 To use this package, follow the following steps
 
 1) Download the original dataset from mendely data (doi: 10.17632/npk2zcm85h.2). Make sure to download version 2 of the dataset which is named 'indoor_fire_detection_multisensornodes_dataset.csv'.
 2) Store the dataset as .csv file with name 'indoor_fire_detection_multisensornodes_dataset.csv' into the following path: '...\D3-fire_detection_framework\model\data'
 3) Run the three core model notebooks: 
- Run [`data_preprocessing.ipynb`](model/data_preprocessing.ipynb) to remove ventilation artefacts, ensure correct data format for further processing as a multivariate time series
- Run the two model notebooks ([`detecting_early_fire_indicator_patterns_single_node.ipynb`](model/detecting_early_fire_indicator_patterns_single_node.ipynb) and [`detecting_early_fire_indicator_patterns_network.ipyn`](model/detecting_early_fire_indicator_patterns_network.ipyn). They will automatically store the .csv files containing the predictions and explanations in the correct folder ('...\D3-fire_detection_framework\model\export\single_node resp. \network'). Note that you can define the options for scalers used in the pipeline in both scripts. The runtime of the model scripts depends on the amount of defined scalers.
 4) Run the explanation app to investiagte the model performance and time-related model explanations:
- Open a terminal running your python env (e.g. CMD prompt when using windows) and navigate to the model folder ('...\D3-fire_detection_framework\model')
- Run [`explanation_module_app.py`](model/explanation_module_app.py) using the command 'streamlit run explanation_module_app.py'. After running that command, a new tab opens automatically in your default web browser displaying the explanation app
 
 ## How to cite
