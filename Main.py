#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
rental = pd.read_csv("daily_rent_detail.csv")
station = pd.read_csv("station_list.csv")
freq = pd.read_csv("usage_frequency.csv")
weather = pd.read_csv("weather.csv")

# %%
print(rental.dtypes)
print(rental.count())
# %%
print(station.dtypes)
print(station.count())
# %%
print(freq.dtypes)
print(freq.count())
# %%
print(weather.dtypes)
print(weather.count())
# %%
# Checking the number of null values in rental data frame
print(rental.isnull().sum())
# %%
 # Percentage of missing values
missing_percentage = rental.isnull().sum() / len(rental) * 100
print(missing_percentage)
print("\n")
# %%
rental.head()
# %%
# Removing all null values from the rental data frame
rental = rental.dropna()

# To reset the index after dropping rows, use:
rental = rental.reset_index(drop=True)
# %%
rental.count() #Contains 13928217 rows after removing null values
# %%
rental['started_at'] = pd.to_datetime(rental['started_at'],format="%Y-%m-%d %H:%M:%S", errors='coerce') # Changing column from object to datetime
rental['ended_at'] = pd.to_datetime(rental['ended_at'],format="%Y-%m-%d %H:%M:%S", errors='coerce')
# %%
rental['start_station_id'] = pd.to_numeric(rental['start_station_id'], errors='coerce').astype('Int64') #Changing Station ID From Object to Integer Data Type
rental['end_station_id'] = pd.to_numeric(rental['end_station_id'], errors='coerce').astype('Int64')

# %%
rental.describe() 

# %%
extra_data = rental[rental['started_at'].isna()]
print(extra_data) # Checking for mismtaches between started_at and ended_at
# %%
rental = rental[rental['started_at'].notna() & rental['ended_at'].notna()]
# %%
#New Column to find total trip duration
rental['trip_duration'] = (rental['ended_at'] - rental['started_at']).dt.total_seconds()/60 # In Minutes

rental['trip_duration'] = rental['trip_duration'].round(2)
# %%
zero_duration_trips = rental[rental['trip_duration'] == 0]

# Display the rows with zero trip duration
print(zero_duration_trips)
# %%
# %%
# %%
# working on the data set weather 
print(weather.info())
print(weather.isnull().sum())
# %%
# delete unnecessary columns 
col_keep= ['name', 'datetime', 'temp', 'humidity', 'precip', 
                   'windgust', 'windspeed', 'visibility', 'conditions', 
                   'description', 'icon']
# %%
# keep neccery columns and create new dataset weather_data
weather_data=weather[col_keep]
#%%
# summery of dataset
print(weather_data.info()
      )
print(weather_data.head())
#%%
# check null values and dtypes 
print(weather_data.isnull().sum()
      )
print(weather_data.isna().sum())
print(weather_data.dtypes)
# %%
#working on the frequency dataset 
# check null values 
print(freq.isnull().sum())
print(freq.isna().sum())
# %%
# check col names for merging these dataset

print("col name of the weather dataset")
print(weather_data.columns)
#%%
print("col name of the freq data set ")
print(freq.columns)
# %%
# merge these 2 dataset as file 
# we take datetime col as common for merging  
weather_data['datetime']=pd.to_datetime(weather_data['datetime'], errors='coerce')
freq['datetime']=pd.to_datetime(freq['datetime'], errors='coerce')
file=pd.merge(weather_data,freq,on='datetime',how='inner')
#%%
# check summery statstics 
print(file.head())
# %%
# check null values 
print(file.isna().sum())
#%%
print(file.info())
# %%
print(file.duplicated().sum())

# %%
