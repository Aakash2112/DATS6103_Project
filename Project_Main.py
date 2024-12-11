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
# Step 3: Check for Missing Values
missing_values = accs.isnull().sum()
print("Missing values per column:\n", missing_values)
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
# %%
# Limit to top 20 categories for each column
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    top_categories = accs[col].value_counts().nlargest(20).index
    sns.countplot(y=accs[accs[col].isin(top_categories)][col], order=top_categories)
    plt.title(f'Frequency of Top 20 Categories in {col}')
    plt.show()
# %%
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
