#%%
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%
# Load the data
accs = pd.read_csv("accidents_final_data.csv")
# %%
accs.head()
# %%
# Step 2: Overview of Data
print("Shape of the dataset:", accs.shape)
print("Columns in the dataset:\n", accs.columns)
print("Summary of data:\n", accs.describe())
print("Columns and data types:\n")
accs.info()
# %%


#%%
# Step 3: Check for Missing Values
missing_values = accs.isnull().sum()
print("Missing values per column:\n", missing_values)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to detect outliers using IQR
def detect_outliers_iqr(accs, column):
    Q1 = accs[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = accs[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
    outliers = accs[(accs[column] < lower_bound) | (accs[column] > upper_bound)]
    return outliers

# Function to remove outliers using IQR
def remove_outliers_iqr(accs, column):
    Q1 = accs[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = accs[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
    accs_cleaned = accs[(accs[column] >= lower_bound) & (accs[column] <= upper_bound)]  # Keep only valid rows
    return accs_cleaned

# List numeric columns in the 'accs' DataFrame
numeric_columns = accs.select_dtypes(include=np.number).columns.tolist()

# Set up the plot size and style
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Plot boxplots for each numeric column to visualize outliers
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=accs, x=column)
    plt.title(f"Box Plot for {column}")
    plt.show()

# Remove outliers from 'Estimated Vehicle Speed'
accs_cleaned = remove_outliers_iqr(accs, 'Estimated Vehicle Speed')

# Check the shape of the dataset after removing outliers
print(f"Original dataset shape: {accs.shape}")
print(f"Cleaned dataset shape: {accs_cleaned.shape}")

# Plot boxplots for each numeric column to visualize outliers in cleaned data
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=accs_cleaned, x=column)
    plt.title(f"Box Plot for {column} (After Removing Outliers)")
    plt.show()
# List of columns to remove outliers
columns_to_check = ['Train Speed', 'Number of Cars', 'Number of Locomotive Units', 'Temperature']

# List numeric columns in the 'accs' DataFrame
numeric_columns = accs.select_dtypes(include=np.number).columns.tolist()

# Set up the plot size and style
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Plot boxplots for each numeric column to visualize outliers before removal
for column in columns_to_check:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=accs, x=column)
    plt.title(f"Box Plot for {column} (Before Removing Outliers)")
    plt.show()

# Remove outliers from the selected columns
accs_cleaned = remove_outliers_iqr(accs, columns_to_check)

# Check the shape of the dataset after removing outliers
print(f"Original dataset shape: {accs.shape}")
print(f"Cleaned dataset shape: {accs_cleaned.shape}")

# Plot boxplots for each numeric column to visualize outliers in cleaned data
for column in columns_to_check:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=accs_cleaned, x=column)
    plt.title(f"Box Plot for {column} (After Removing Outliers)")
    plt.show()

# %%
# Step 4: Univariate Analysis
numerical_columns = accs.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = accs.select_dtypes(include=['object']).columns
# %%
# Numerical data distributions
for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(accs[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

#%%
# Numerical data distributions after removing outliers
for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(accs_cleaned[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

#%% [markdown]

## Visualization

# number of accidents reported in years
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data: replace 'accs' with your actual DataFrame name
# 'Report Year' should be the column in your DataFrame indicating the year of the report

# Set the figure size before creating the plot
plt.figure(figsize=(10, 6))

# Create a count plot for the number of incidents per year with a custom color palette
sns.countplot(data=accs, x='Report Year', palette='plasma')

# Add title and labels
plt.title('Number of Incidents per Year')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Show the plot
plt.show()


#number of accident reported in states
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data: replace 'accs' with your actual DataFrame name
# 'State Name' should be the column in your DataFrame indicating the state where the incident occurred

# Set the figure size before creating the plot
plt.figure(figsize=(15, 8))

# Create a count plot for the number of incidents per state with a custom color
sns.countplot(data=accs, x='State Name', order=accs['State Name'].value_counts().index, palette='viridis')

# Add title and labels
plt.title('Number of Incidents per State')
plt.xlabel('State')
plt.ylabel('Number of Incidents')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Show the plot
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(accs['Temperature'], bins=100, kde=True, color='blue')
plt.xlim(-30, 150)
plt.title('Distribution of Temperature')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()

#impact of different weather conditions on incident rates
import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size before creating the plot
plt.figure(figsize=(12, 8))

# Create a count plot for the number of incidents by weather condition
sns.countplot(data=accs, x='Weather Condition', palette='Set1')

# Add title and labels
plt.title('Number of Incidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Incidents')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()
#visibility conditions correlate with incidents.
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
visibility_counts = accs['Visibility'].value_counts()

# Create a pie chart with a different color palette
plt.figure(figsize=(10, 7))
plt.pie(visibility_counts, labels=visibility_counts.index, autopct='%1.1f%%', colors=sns.color_palette('coolwarm'))

# Add title
plt.title('Distribution of Incidents by Visibility Condition')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot for estimated vehicle speed and train speed
plt.figure(figsize=(12, 8))
sns.scatterplot(data=accs, x='Estimated Vehicle Speed', y='Train Speed', color='purple')

# Set the x-axis and y-axis limits to the specified range
plt.xlim(0, 110)
plt.ylim(0, 110)

# Add title and labels
plt.title('Correlation Between Estimated Vehicle Speed and Train Speed During Incidents')
plt.xlabel('Estimated Vehicle Speed')
plt.ylabel('Train Speed')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Prepare the pivot table for heatmap
visibility_weather_pivot = accs.pivot_table(index='Weather Condition', columns='Visibility', values='Incident Number', aggfunc='count')

# Set the figure size
plt.figure(figsize=(14, 10))

# Create a heatmap
sns.heatmap(visibility_weather_pivot, cmap='YlGnBu', annot=True, fmt='d')

# Add title and labels
plt.title('Heatmap of Incidents by Weather Condition and Visibility')
plt.xlabel('Visibility Condition')
plt.ylabel('Weather Condition')

# Show the plot
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size before creating the plot
plt.figure(figsize=(14, 8))

# Create a count plot for highway user actions by different types of highway users
sns.countplot(data=accs, x='Highway User Action', hue='Highway User', palette='Set2')

# Add title and labels
plt.title('Incidents by Highway User Actions and Types of Highway Users')
plt.xlabel('Highway User Action')
plt.ylabel('Number of Incidents')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size before creating the plot
plt.figure(figsize=(14, 8))

# Create a box plot for train speeds by track type
sns.boxplot(data=accs, x='Track Type', y='Train Speed', palette='coolwarm')

# Set the y-axis limit to the specified range
plt.ylim(0, 120)

# Add title and labels
plt.title('Train Speeds by Track Type During Incidents')
plt.xlabel('Track Type')
plt.ylabel('Train Speed')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size before creating the plot
plt.figure(figsize=(14, 8))

# Create a count plot for the number of incidents by track type
sns.countplot(data=accs, x='Track Type', palette='Set2')

# Add title and labels
plt.title('Number of Incidents by Track Type')
plt.xlabel('Track Type')
plt.ylabel('Number of Incidents')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size before creating the plot
plt.figure(figsize=(14, 8))

# Create a count plot for vehicle direction by train direction
sns.countplot(data=accs, x='Vehicle Direction', hue='Train Direction', palette='Set2')

# Add title and labels
plt.title('Incidents by Vehicle Direction and Train Direction')
plt.xlabel('Vehicle Direction')
plt.ylabel('Number of Incidents')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()

# Step 5: Correlation Analysis
correlation_matrix = accs[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()



# %%
# Step 6: Feature Engineering
# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_columns:
    accs[col] = label_encoder.fit_transform(accs[col])

# Scale numerical variables
scaler = StandardScaler()
accs[numerical_columns] = scaler.fit_transform(accs[numerical_columns])
# %%
# Step 7: Train-Test Split
if 'Vehicle Damage Cost' in accs.columns:
    X = accs.drop(columns=['Vehicle Damage Cost'])
    y = accs['Vehicle Damage Cost']
else:
    raise ValueError("Target variable 'Vehicle Damage Cost' not found in the dataset.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Save processed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("EDA and preprocessing completed. Data is ready for modeling.")
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure your 'accs' dataframe is loaded properly
# accs = pd.read_csv('your_data.csv')  # Uncomment if not already loaded

# Set up the plot style
sns.set(style="whitegrid")

# 2. Weather and Year
plt.figure(figsize=(10, 6))
sns.countplot(data=accs, x='Report Year', hue='Weather Condition')
plt.title('Weather Condition by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 3. Visibility and Weather
plt.figure(figsize=(10, 6))
sns.boxplot(data=accs, x='Weather Condition', y='Visibility')
plt.title('Visibility by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Visibility')
plt.xticks(rotation=45)
plt.show()



#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'Weather Condition' is encoded as integers, let's create a mapping to their string labels
weather_mapping = {
    0: 'Clear',
    1: 'Cloudy',
    2: 'Snow',
    3: 'Rain',
    4: 'Fog',
    5: 'Sleet'
}

# Map the numerical weather condition to the corresponding name
accs['Weather Condition'] = accs['Weather Condition'].map(weather_mapping)

# Set up the plot style
sns.set(style="whitegrid")

# Plot the count of Weather Conditions by Year
plt.figure(figsize=(10, 6))
sns.countplot(data=accs, x='Report Year', hue='Weather Condition')

# Add the title and labels
plt.title('Weather Condition by Year')
plt.xlabel('Year')
plt.ylabel('Count')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.show()
#%%