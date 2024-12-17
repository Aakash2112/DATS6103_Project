# US Highway Railroad - Crossing Accidents Analysis & Prediction

# DATA SOURCE:
URL: https://www.kaggle.com/datasets/yogidsba/us-highway-railgrade-crossing-accident/data

## INTRODUCTION
The dataset titled Highway_railroad_accidents.csv from kaggle contained:
1) 246000 rows
2) 141 Columns

This dataset was extensively preprocessed and underwent feature selection prior to being used for our smart questions (located in preprocess.py).
Some of the prominent columns included:
1) Crossing Warning Location
2) Weather Condition
3) Train Speed etc.

# LIMTATIONS OF THE DATASET:
1) There were few errors or inconsistencies noticed in reported metrics like train speeds and train damage.
2) The raw data required extensive cleaning for analysis due to it's manual entry.
3) The environmental factors in the dataset are recorded for the day of the accident but do not account for the specific time it occurred.
   
# SMART QUESTIONS:
1) How can we predict the severity of a driver's injury in a railroad crossing accident using external factors?
2) How can we identify accident-prone locations in USA based on accident frequency over the past 46 years?
3) How can we predict the location of crossing warning sign present during railroad accidents, based on historical data?

# Additional Python Libraries :
1) imblearn - https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
2) plotly - https://plotly.com/

# Group Members:
1) Aakash Hariharan
2) Vishal Fulsundar
3) Abhilasha Singh
4) Trisha Singh





