#%%
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# %%
data = pd.read_csv('Highway-Rail_Grade_Crossing_Accident_Data.csv')
# %%
# 
print("Dataset shape - ", data.shape)
print("dataset first 5 rows - ",data.head())
print("dataset describe -",data.describe()
      )
print("dataset columns and Types  -",data.info()
      )
#%%
# check missing values and handle values 
missing_values = data.isnull().sum()
print(" missing values per col:\n",missing_values)

#%%
unique_values = data.nunique()
print(unique_values)

# %%
print(" the shape of the dataset now: ",data.shape)
missing_values = data.isnull().sum()

# Print the column names and their respective missing values
for column, missing in missing_values.items():
    print(f"{column}: {missing}")
#%%
column_counts = data.count()
#%%
print("shape of the dataset",data.shape)
#%%
# Print the column names and their respective counts
for column, count in column_counts.items():
    print(f"{column}: {count}")

# %%
threshold = 100000
clean_data = data.dropna(thresh=len(data) - threshold, axis=1)
print("the shape of the dataset-",clean_data.shape)
#%%

missing_values = clean_data.isnull().sum()
print(missing_values)
#%%
clean_data=clean_data.dropna()

print("shape of the dataset- ", clean_data.shape)
#%%
import os
print(os.getcwd())
clean_data
#%%
# Print the shape of the cleaned dataset
print("Shape of the dataset: ", clean_data.shape)

# %%
columns_to_delete = [
    "Crossing Users Injured For Reporting Railroad",
    "Crossing Users Killed For Reporting Railroad",
    "incident number",
    "incident year",
    "incident month",
    "Maintainance Incident Number",
    "Maintenance Incident Year",
    "Maintenance Incident Month",
    "Grade Crossing ID",
    "Hour",
    "Highway User Code",
    "Vehicle Direction Code",
    "Highway User Position Code",
    "Equipment Involved Code",
    "Railroad Car Unit Position",
    "Equipment Struck Code",
    "Hazmat Involvement Code",
    "Hazmat Involvement",
    "Visibility Code",
    "Weather Condition Code",
    "Equipment Type Code",
    "Track Type Code",
    "Track Class",
    "Estimated/Recorded Speed",
    "Train Direction Code",
    "Crossing Warning Expanded Code 12",
    "Crossing Warning Expanded 12",
    "Crossing Warning Location Code",
    "Crossing Illuminated",
    "Highway User Action Code",
    "View Obstruction Code",
    "Driver Condition Code",
    "Crossing Users Killed For Reporting Railroad",
    "Crossing Users Injured For Reporting Railroad",
    "Employees Killed For Reporting Railroad",
    "Employees Injured For Reporting Railroad",
    "Form 54 Filed",
    "Passengers Killed For Reporting Railroad",
    "Passengers Injured For Reporting Railroad",
    "Total Killed Form 57",
    "Total Injured Form 57",
    "Railroad Type",
    "Joint Code",
    "Total Killed Form 55A",
    "Total Injured Form 55A",
    "District",
    "Report Key",
    "Day",
    "Minute",
    "AM/PM",
]

clean_data.columns = clean_data.columns.str.strip()
columns_to_delete_cleaned = [col.strip() for col in columns_to_delete]

clean_data = clean_data.drop(columns=[col for col in columns_to_delete_cleaned if col in clean_data.columns])
#%%
print(clean_data)
print(" the shape of the data now - ", clean_data.shape)
# %%

#clean_data.to_csv('accidents_final_data.csv', index=False)


# %%
