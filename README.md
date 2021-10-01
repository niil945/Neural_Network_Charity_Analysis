# Neural Network Charity Analysis

## Overview

In this project we utilize neural networks to predict whether charity organizations applying for funding from Alphabet Soup will be
based on a dataset that includes details from Alphabet Soup of previous charity organizations that received funds, including whether such
organizations were successful or not, how much funding they requested and received, and other miscellaneous information. Using these
features we build models to predict which future applications will be successful. The target goal for the model was a 75+% accuracy rate
in predicting success.

## Data

The dataset consisted of 34,299 rows and included the following colums:
 - EIN
 - NAME
 - AFFILIATION
 - CLASSIFICATION
 - USE_CASE
 - ORGANIZATION
 - STATUS
 - INCOME_AMT
 - SPECIAL_CONSIDERATIONS
 - ASK_AMT
 - IS_SUCCESSFUL

## Results

### Data Pre-Processing

An assessment of the data provided insight on required pre-processing and and understanding of the data.

 - The target variable is the IS_SUCCESSFUL column.
 - The EIN and NAME columns were dropped as not being relevant to the prediction of success.
 - All other columns were features of the initial methodology.
 - APPLICATION_TYPE was binned upon the count of each instance, with each value of less than 1,000 occurrences being combined into an 'Other' category.
 - CLASSIFICATION was binned based upon the count of each instance, with each value with less than 1,000 occurrences being combined into an 'Other' category.
 - Columns of dtype 'Object' were encoded with OneHotEncoder.
 - Columns were scaled using StandardScaler.

### Compiling, Training, and Evaluating the Model

 - Base Model  
![Base Model Summary](/Resources/Images/base_model_summary.png)
![Base Model Epoch 100](/Resources/Images/base_model_epoch100.png)
 - Optimization Attempt 1
   - Variance from base model: Added a third layer, changed nodes (80, 40, 20)
![Optimization Model 1 Summary](/Resources/Images/opt1_model_summary.png)
![Optimization Model 1 Epoch 100](/Resources/Images/opt1_model_epoch100.png)
- Optimization Attempt 2
   - Variance from base model: Added a third layer, changed nodes (100, 65, 15), used 250 epochs
![Optimization Model 2 Summary](/Resources/Images/opt2_model_summary.png)
![Optimization Model 2 Epoch 100](/Resources/Images/opt2_model_epoch250.png)
 - Optimization Attempt 3
   - Variance from base model: changed nodes (80, 60), changed activation on first hidden layer to 'tanh', used 250 epochs
![Optimization Model 3 Summary](/Resources/Images/opt3_model_summary.png)
![Optimization Model 3 Epoch 100](/Resources/Images/opt3_model_epoch250.png)
 
No model achieved the target goal of 75+% accuracy.

## Summary

Overall I had very little success in varying nodes, activations, or epochs that resulted in any meaningful change to the resulting success of the neural
network. In addition to the models above I tried to remove columns that seemed to have no correlation to the outcome (AFFILIATION and ORGANIZATION) but found
the accuracy dropped to ~62%, a significant drop from the base model. In a prediction model the expectation is that it's not realistic to expect 100% accuracy,
however I did expect that efforts to optimize the model would result in meaningful change. This leads me to believe that the factors that contribute to success
of charity organizations lies outside the scope of the data captured in the dataset. In this scenario I would advise that projects be assessed to determine
what additional information could be captured to include in models in the future to provide a more accurate prediction of outcome. Considering the accuracy
declined when excluding AFFILIATION and ORGANIZATION columns, I would posit that it's not always beneficial to consider past successes to determine potential
future successes. That each charity application be considered upon its own merits. Otherwise this could lead to predictions biased in favor of an application solely
on the basis that previous applications by the same organization or affiliate were successful. 

## Resources
 - Data Source: [Raw CSV data file](https://raw.githubusercontent.com/niil945/Neural_Network_Charity_Analysis/main/Resources/charity_data.csv)
 - Software: Python 3.7, Jupyter Notebook
 - Libraries: os, pandas, sklearn, tensorflow
