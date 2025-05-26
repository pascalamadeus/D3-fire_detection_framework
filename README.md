# D3-fire_detection_framework
 This repository contains the code used in the paper "Enhancing Early Indoor Fire Detection Using Indicative Patterns in Multivariate Time Series Data Based on Multi-Sensor Nodes"
 
 ## Overview
 
 
 
 
 ## Abstract
 
 Multi-sensor technology is becoming increasingly accessible, resulting in more
complex data structures that require reasonable analysis in order to extract crucial information. In this study, we propose a novel approach for enhancing early indoor fire detection using multi-sensor nodes. We model the task of early indoor fire detection as a binary classification problem in multivariate time series (MTS) data to capture early fire indicator patterns in their (a) dimension of emissions, (b) dimension of sensor placement and (c) dimension of time. We combine a prediction component based on ROCKET (RandOm Convolutional KErnel Transformation) and an explanation component based on SHAP (SHapley Additive exPlanations) as a novel approach to provide robust early fire detection and model agnostic explanations simultaneously in one system. Based on five different sensor measurements (CO, H2, VOC (volatile organic compounds), PM05, and PM10 (particulate matter)) mea-sured at nine different sensor node positions, our results highlight VOC and PM as the main early fire indicators, outperforming gases like H2 or CO. We demonstrate that the dimension of time is just as important to early indoor fire detection as the absolute sensor measurement value (dimension of emissions). Our findings indicate that a single node approach is more suitable than a network approach for the purpose of early indoor fire detec-tion. Furthermore, we could confirm the presence of numerical underflow in the KernalSHAP explanation for MTSC, which significantly reduces SHAP values (by up to 61 \%) at longer interval lengths. This effect also depends on the total number of input features.
 
 ## Usage
 
 
 ## Setup
 
 
 
 ## Dataset
 


## How to cite