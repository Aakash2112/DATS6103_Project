
#%% Import Libraries
import numpy as np
#%%
import pandas as pd
import pandas as pd
print(pd.__version__)
#%%
#%% 
accs = pd.read_csv("accidents_final_data.csv")
#%% Data Inspection
accs.info()  
accs.describe()  
#%%
accs.info()
accs.describe()
missing_values = accs.isnull().sum()
print(missing_values)
# %%
duplicates = accs.duplicated()
print(accs[duplicates])
print(accs.shape[0])
# %%
accs= accs.drop_duplicates()
print(accs.shape[0])
#%%
# Inspect the columns and data types
accs.info()
#%%
# View unique values in each column to spot categorical columns
for column in accs.columns:
    print(f"Column: {column}")
    print(accs[column].unique())
    print("="*50)   
import copy
accs2=copy.deepcopy(accs)
accs2

#%%
accs2
accs2.head()
from sklearn.preprocessing import LabelEncoder

# Convert categorical columns to numeric using LabelEncoder
categorical_columns = accs2.select_dtypes(include=['object']).columns

encoder = LabelEncoder()
for col in categorical_columns:
    accs2[col] = encoder.fit_transform(accs2[col].astype(str))

# Recalculate the correlation matrix
corr_matrix = accs2.corr()

# Plot heatmap
plt.figure(figsize=(40, 30))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()



#%%

# %%
# Boxplot for outliers
for col in ['Train Speed', 'Temperature', 'Vehicle Damage Cost']:
    sns.boxplot(x=accs2[col])
    plt.title(f"Outlier Detection in {col}")
    plt.show()

# %%
# Filling numerical missing values with the median
num_cols = accs2.select_dtypes(include=['float', 'int']).columns
accs2[num_cols] = accs2[num_cols].fillna(accs2[num_cols].median())

# Filling categorical missing values with the mode
cat_cols = accs2.select_dtypes(include=['object']).columns
for col in cat_cols:
    accs2[col] = accs2[col].fillna(accs2[col].mode()[0])

# Verify no missing values remain
print(accs2.isnull().sum())


# %%
# Inspect unique values in categorical columns
for col in accs2.select_dtypes(include=['object']).columns:
    print(f"Unique values in {col}:\n{accs2[col].unique()}\n")

# Validate numerical columns for outliers
print(accs2.describe())

# %%
def classify_severity(damage_cost):
    if damage_cost > 10000:
        return 'High'
    elif damage_cost > 5000:
        return 'Medium'
    else:
        return 'Low'

accs2['Incident Severity'] = accs2['Vehicle Damage Cost'].apply(classify_severity)

# %%
def calculate_risk(weather, visibility, train_speed):
    if weather in ['Rain', 'Snow'] or visibility in ['Dark'] or train_speed > 60:
        return 'High Risk'
    else:
        return 'Low Risk'

accs2['Risk Factor'] = accs2.apply(
    lambda x: calculate_risk(x['Weather Condition'], x['Visibility'], x['Train Speed']), axis=1
)

# %%
# Distribution plots
for col in ['Train Speed', 'Temperature', 'Vehicle Damage Cost']:
    sns.histplot(accs2[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# %%
sns.countplot(y='Weather Condition', data=accs2, order=accs2['Weather Condition'].value_counts().index)
plt.title("Weather Condition Distribution")
plt.show()

# %%
sns.boxplot(x='Incident Severity', y='Train Speed', data=accs2, palette='viridis')
plt.title("Train Speed by Incident Severity")
plt.show()

# %%
sns.violinplot(x='Risk Factor', y='Vehicle Damage Cost', data=accs2, palette='muted')
plt.title("Vehicle Damage Cost by Risk Factor")
plt.show()

# %%
for col in ['Train Speed', 'Temperature', 'Vehicle Damage Cost']:
    sns.boxplot(x=accs2[col])
    plt.title(f"Outlier Detection for {col}")
    plt.show()

# %%
# Monthly incident trends
monthly_incidents = accs2.groupby('Incident Month').size()
monthly_incidents.plot(kind='bar', color='skyblue')
plt.title("Monthly Incident Trends")
plt.xlabel("Month")
plt.ylabel("Number of Incidents")
plt.show()
# %%
state_incidents = accs2['State Name'].value_counts()

# Simple bar plot for states
state_incidents.plot(kind='bar', figsize=(12, 6), color='teal')
plt.title("Number of Incidents by State")
plt.xlabel("State")
plt.ylabel("Number of Incidents")
plt.show()

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare data for clustering
cluster_data = accs2[['Train Speed', 'Vehicle Damage Cost', 'Temperature']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
accs2['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize clusters
sns.scatterplot(x='Train Speed', y='Vehicle Damage Cost', hue='Cluster', data=accs2, palette='viridis')
plt.title("Incident Clusters")
plt.show()

#%%
# Summary table
summary = accs2.groupby('Incident Severity').agg({
    'Vehicle Damage Cost': ['mean', 'median'],
    'Train Speed': ['mean', 'max'],
    'Temperature': ['mean']
})
print(summary)


# %%

# Set consistent color palette and figure size
sns.set_palette(sns.color_palette("PuRd", 8))
plt.figure(figsize=(15, 13))
sns.set_palette(sns.color_palette("Set2", 8))
plt.figure(figsize=(15, 13))

# Distribution plot for Report Year
sns.histplot(data=accs2, x='Report Year', kde=True, bins=15, color='blue')
plt.title("Distribution of Report Year", fontsize=16)
plt.xlabel("Report Year", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

# %%
# 2nd map brekthrough 
print(accs['State Name'].unique())

accs2
# %%
# Assuming accs dataset is already loaded
# Step 1: Check the format of state names and capitalize
accs['State Name'] = accs['State Name'].str.capitalize()

# Step 2: Aggregate data by state to get count of accidents
df_state = accs.groupby(['State Name']).count()['Report Year'].sort_values(ascending=False).reset_index()
df_state.rename(columns={'Report Year': 'Count_of_accident'}, inplace=True)

# Step 3: Map state names to state codes
code = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Map the state names to their respective codes
df_state['Code'] = df_state['State Name'].map(code)

# Step 4: Check for missing values in state codes (if any)
missing_states = df_state[df_state['Code'].isnull()]
if not missing_states.empty:
    print(f"Missing state codes: {missing_states}")
    df_state = df_state.dropna(subset=['Code'])  # Drop rows with missing state codes

# Step 5: Create the choropleth map
fig = px.choropleth(df_state,
                    locations='Code',  # State abbreviations
                    locationmode='USA-states',  # USA states mode
                    color='Count_of_accident',  # Use accident count for coloring
                    hover_name='State Name',  # Show state names on hover
                    title='State-Wise Incident Report',
                    color_continuous_scale='sunset',
                    scope='usa')

# Show the map
fig.show()
#%%