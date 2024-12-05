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
rental.dtypes
rental.count()
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
print(rental.isnull().sum())
# %%
