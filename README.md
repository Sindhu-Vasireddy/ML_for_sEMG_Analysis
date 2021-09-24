# ML_for_sEMG_Analysis

This repository contains MATLAB code files and data necessary to perform a comparative study of classical and advanced machine learning methods for regression and classification tasks on NinaPro Datasets using surface electromyography (s-EMG) signals. The code has been developed and tested with MATLAB 2021b and requires the following prerequisites to be installed:

- [ ] MATLAB 2021b
- [ ] Deep Learning Toolbox
- [ ] Statistics and Machine Learning Toolbox
- [ ] Parallel Computing Toolbox

## GPU Capability
To enhance performance, you can utilize GPU capability by modifying line 49 in the lstm_rgrn file from 'cpu' to 'gpu'.

## Execution Instructions
Follow these steps to run the code:

- **Download the Files**: Clone or download this repository to your local machine.

- **Feature Extraction**: Run TB_FeatureExtraction.m code file to perform feature extraction.

- **AutoRegression**: Execute AutoRegression.m code file to perform ARX and NARX Regression. If you want to clean the command window and plots, use clc and close all commands if needed. This code runs for all 6 input-output combinations.

- **Linear Regression Models**: Run LS_Regression.m code file to perform Linear Regression models. If you want to clean the command window and plots, use clc and close all commands if needed. This code runs all linear regression models for inputs (v) and outputs (q) specified in line 10 and 11 (default v=12, q=1).

- **Classification**: Execute Classification.m code to perform K-Nearest Neighbors (KNN) and Long Short-Term Memory (LSTM) classification.

- **Generative Adversarial Network (GAN)**: Run gan.m code to train and test a GAN. Lines 5-7 and 37-43 specify the training options for the generator and discriminator, respectively.

## Folder Structure
- [MATLABcodes_withDatafile](./MATLABcodes_withDatafile): Contains standalone executables of MATLAB code files.
- [Plots_and_Outputs](./Plots_and_Outputs): Contains output plots collected from the code execution, including a print of the MATLAB command window for all cases.

## Dissertation Report
A [dissertation report in PDF format](Dissertation_Report.pdf) is also included, presenting this work as part of the MSc in Advanced Control and Systems Engineering at the University of Sheffield, submitted in 2021.