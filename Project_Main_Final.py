#%%
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

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
# Univariate Analysis
numerical_columns = accs.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = accs.select_dtypes(include=['object']).columns

## Visualization
# Distribution plot for Report Year
sns.histplot(data=accs, x='Report Year', kde=True, bins=15, color='blue')
plt.title("Distribution of Report Year", fontsize=16)
plt.xlabel("Report Year", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

#In 1980, the incident count was significantly higher, but over the years, there has been a sharp decline in the number of incidents.

# Distribution plot for Incident Month
sns.histplot(data=accs, x='Incident Month', kde=True, bins=12, color='blue')  # Adjusted for months
plt.title("Distribution of Incidents by Month", fontsize=16)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

#January and December show the highest number of incidents, while April, May, and June experience a small drop of around 3,000 incidents.
#%%
#Number of accident reported in states
plt.figure(figsize=(15, 8))
sns.countplot(data=accs, x='State Name', order=accs['State Name'].value_counts().index, color='seagreen')
plt.title('Number of Incidents per State')
plt.xlabel('State')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=90)
plt.show()

#Texas has the significantly highest number of incidents, followed by Illinois, Indiana. In contrast, Hawaii and District of Columbia has almost no incidents.
#%%
#Distribution of Temperature
plt.figure(figsize=(10, 6))
sns.histplot(accs['Temperature'], bins=100, kde=True, color='lightblue')
plt.xlim(-30, 150)
plt.title('Distribution of Temperature')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()

#Temperature is normally distributed.
#%%
#Number of Incidents by Weather Condition
plt.figure(figsize=(12, 8))
sns.countplot(data=accs, x='Weather Condition')
plt.title('Number of Incidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.show()

#Clear weather is linked to a significant number of incidents

#visibility conditions correlate with incidents.
visibility_counts = accs['Visibility'].value_counts()
plt.figure(figsize=(10, 7))
plt.pie(visibility_counts, labels=visibility_counts.index, autopct='%1.1f%%', colors=sns.color_palette('coolwarm'))
plt.title('Distribution of Incidents by Visibility Condition')
plt.show()

#visibility graph shows that 50% of incidents occur during daylight hours, followed by 38% during nighttime.

#Heatmap of Incidents by Weather Condition and Visibility
visibility_weather_pivot = accs.pivot_table(index='Weather Condition', columns='Visibility', values='Incident Number', aggfunc='count')
plt.figure(figsize=(14, 10))
sns.heatmap(visibility_weather_pivot, cmap='YlGnBu', annot=True, fmt='d')
plt.title('Heatmap of Incidents by Weather Condition and Visibility')
plt.xlabel('Visibility Condition')
plt.ylabel('Weather Condition')
plt.show()

#The heatmap also highlights a higher number of incidents occurring under clear skies and during daytime.
#%%
# Create a scatter plot for estimated vehicle speed and train speed
plt.figure(figsize=(12, 8))
sns.scatterplot(data=accs, x='Estimated Vehicle Speed', y='Train Speed', color='seagreen')
plt.xlim(0, 110)
plt.ylim(0, 110)
plt.title('Correlation Between Estimated Vehicle Speed and Train Speed During Incidents')
plt.xlabel('Estimated Vehicle Speed')
plt.ylabel('Train Speed')
plt.show()

#The majority of accidents occurred when vehicle speeds were below 20, while train speeds ranged between 0 and 60.
#%%
#plot for highway user actions by different types of highway users
plt.figure(figsize=(14, 8))
sns.countplot(data=accs, x='Highway User Action', hue='Highway User',palette='Set2')
plt.title('Incidents by Highway User Actions and Types of Highway Users')
plt.xlabel('Highway User Action')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.show()

#The graph shows that the majority of incidents occurred when the vehicle did not stop, with autos being the most common type involved in the collisions.
#%%
#Train Speeds by Track Type During Incidents
plt.figure(figsize=(14, 8))
sns.boxplot(data=accs, x='Track Type', y='Train Speed', color='green')
plt.ylim(0, 120)
plt.title('Train Speeds by Track Type During Incidents')
plt.xlabel('Track Type')
plt.ylabel('Train Speed')
plt.xticks(rotation=45)
plt.show()
#The train speed on the main track typically ranges between 10 and 40, whereas on other rail tracks, the speed is generally below 10, with a few outliers.

#Plot for the number of incidents by track type
plt.figure(figsize=(14, 8))
sns.countplot(data=accs, x='Track Type', color='seagreen')
plt.title('Number of Incidents by Track Type', fontsize=16)
plt.xlabel('Track Type', fontsize=12)
plt.ylabel('Number of Incidents', fontsize=12)
plt.xticks(rotation=45)
plt.show()

#A significantly higher number of incidents have been reported on the main track compared to other track types.

#Plot for vehicle direction by train direction
plt.figure(figsize=(14, 8))
sns.countplot(data=accs, x='Vehicle Direction', hue='Train Direction', palette='Set2')
plt.title('Incidents by Vehicle Direction and Train Direction')
plt.xlabel('Vehicle Direction')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.show()

#The graph shows that there are more incidents when the train's direction is 90 degrees compared to when it is 180 degrees

# Correlation Analysis
correlation_matrix = accs[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()



# %%
#SMART Q1 : How can we predict the severity of a driver's injury in a railroad crossing accident using external factors?
#%%
#EDA
df = pd.read_csv("accidents_final_data.csv") #Declared just for EDA
# Categorical features vs Injury Severity
sns.countplot(x='Highway User', hue='Driver Condition', data=df)
plt.title('User vs Injury Severity')
plt.xticks(rotation=45, ha='right')
plt.show()

#%%
# Count occurrences
severity_by_weather = df.groupby(['Weather Condition', 'Driver Condition']).size().unstack(fill_value=0)

pastel_colors = ['#AEC6CF', '#FFB347', '#77DD77']
# Bar Plot
severity_by_weather.plot(kind='bar', stacked=True, figsize=(10, 6),color=pastel_colors)
plt.title('Injury Severity by Weather Condition')
plt.ylabel('Count')
plt.xlabel('Weather Condition')
plt.legend(title='Injury Severity')
plt.show()

#%%
killed_df = df[df['Driver Condition'] == 'Killed']

# Now, aggregate by 'state' to count the number of 'killed' drivers
state_killed_counts = killed_df['State Name'].value_counts().reset_index()
state_killed_counts.columns = ['State Name', 'Killed_count']

# Sort the values to see the states with the most fatalities
state_killed_counts = state_killed_counts.sort_values(by='Killed_count', ascending=False)

# Plot the data
plt.figure(figsize=(14, 8))
sns.barplot(x='Killed_count', y='State Name', data=state_killed_counts, palette='viridis')

#%%
# Sort the DataFrame by 'Killed_count' in descending order and select the top 10
top_10_states = state_killed_counts.sort_values('Killed_count', ascending=False).head(10)

# Plot the pie chart for the top 10 states
top_10_states.set_index('State Name')['Killed_count'].plot(kind='pie', autopct='%1.1f%%', figsize=(10, 8), cmap='viridis', legend=False)

# Add title
plt.title('Proportion of Killed Drivers by States (Top 10)', fontsize=16)
plt.ylabel('')  # Remove the ylabel
plt.show()

#%%
accs1 = accs # Declared for sake of modeling
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
categorical_columns = accs1.select_dtypes(include=['object']).columns

# Encode all categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    accs1[col] = le.fit_transform(accs1[col])
    label_encoders[col] = le

#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#%%

X = accs1.drop('Driver Condition', axis=1)  # Features: exclude the target column
y = accs1['Driver Condition'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
# MODEL 1: DECISION TREE

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)


#%%
from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(X_train1, y_train1)  # y_train is multi-class


# %%
y_pred1 = model_tree.predict(X_test1)

# Evaluate the model
accuracy = accuracy_score(y_test1, y_pred1)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
print(classification_report(y_test1, y_pred1))

# %%
from sklearn.model_selection import GridSearchCV

# Define a parameter grid to tune the model
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Create a DecisionTreeClassifier instance
dt_model = DecisionTreeClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the model on the training data
grid_search.fit(X_train1, y_train1)

# Print best hyperparameters
print("Best hyperparameters found: ", grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test1)
print("Best Model Accuracy:", accuracy_score(y_test1, y_pred_best))

# %%
report_tree = classification_report(y_test1, y_pred_best)
print("Classification Report:\n", report_tree)
# %%

# %%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=X.columns, class_names=[str(cls) for cls in y.unique()], rounded=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Decision Tree ROC Curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Binarize labels for multiclass
y_bin = label_binarize(y_test1, classes=[0, 1, 2])  # Adjust for your classes
y_score = best_model.predict_proba(X_test1)

# Plot ROC curve for each class
for i in range(y_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc='lower right')
plt.show()

# %%
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X, y, cv=5, n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Cross-validation score')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
# Compute the confusion matrix
cm = confusion_matrix(y_test1, y_pred_best)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Killed', 'Injured', 'Uninjured'], yticklabels=['Killed', 'Injured', 'Uninjured'], cbar=False)

# Add labels and title
plt.title('Confusion Matrix for Decision Tree (Multiclass)', fontsize=14)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
#%%
#MODEL 2: Random Forest Classifier
# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)  

#%%
# Make predictions on the test data
y_pred = rf_model.predict(X_test)  

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

#%%
# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display the feature names and their importance scores
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the sorted feature importance
print(importance_df)

#%%
import matplotlib.pyplot as plt

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to display most important features at the top
plt.show()

#%%
threshold = 0.01
low_importance_features = importance_df[importance_df['Importance'] < threshold]['Feature']

# Drop the low-importance features from the dataframe
X_reduced = X.drop(columns=low_importance_features)

print("Dropped features with low importance:", low_importance_features)
#%%
# Re-train the model with reduced features
X_train, X_test, y_train1, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
model_reduced = RandomForestClassifier(random_state=42)
model_reduced.fit(X_train, y_train1)

# Evaluate the model on the test set
accuracy = model_reduced.score(X_test, y_test)
print(f"Model accuracy after removing low-importance features: {accuracy:.4f}")
#%%

y_pred = model_reduced.predict(X_test)
#accuracy = accuracy_score(X_test, y_test)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

#%%
# Feature Importance After Feature Selection
feature_importances = model_reduced.feature_importances_

# Create a DataFrame to display the feature names and their importance scores
feature_df = pd.DataFrame({
    'Feature': X_reduced.columns,
    'Importance': feature_importances
})

# Sort the features by importance
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to display most important features at the top
plt.show()
#%%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Binarize the output labels for multiclass
y_train_bin = label_binarize(y_train1, classes=[0, 1, 2])  # Adjust classes as per your dataset
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Initialize the RandomForestClassifier with OneVsRestClassifier
model_reduced1 = OneVsRestClassifier(RandomForestClassifier(random_state=42))

# Fit the model on the training data
model_reduced1.fit(X_train, y_train_bin)

# Predict the probabilities for the test set
y_prob = model_reduced1.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_prob.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(y_prob.shape[1]), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass')
plt.legend(loc="lower right")
plt.show()

#%%
cm1 = confusion_matrix(y_test, y_pred)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm1, annot=True, fmt='d', cmap='viridis', xticklabels=['Killed', 'Injured', 'Uninjured'], yticklabels=['Killed', 'Injured', 'Uninjured'], cbar=False)

# Add labels and title
plt.title('Confusion Matrix for Random Forest', fontsize=14)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)

#%%
# SMART Q2: How can we identify accident-prone locations in USA based on accident frequency over the past 46 years?
#%%

#%%
import plotly.express as px
# Create a US Map to plot number of accidents state wise
#The format of state names and capitalize
accs['State Name'] = accs['State Name'].str.capitalize()
#%%
# Aggregate data by state to get count of accidents
df_state = accs.groupby(['State Name']).count()['Report Year'].sort_values(ascending=False).reset_index()
df_state.rename(columns={'Report Year': 'Count_of_accident'}, inplace=True)
#%%
# Map state names to state codes
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
#%%
# Map the state names to their respective codes
df_state['Code'] = df_state['State Name'].map(code)
#%%
# Check for missing values in state codes (if any)
missing_states = df_state[df_state['Code'].isnull()]
if not missing_states.empty:
    print(f"Missing state codes: {missing_states}")
    df_state = df_state.dropna(subset=['Code'])  # Drop rows with missing state codes
    
#%%
# Create the choropleth map    
fig = px.choropleth(df_state, #Requires nbformat>=4.2.0 to run this code
                    locations='Code',  # State abbreviations
                    locationmode='USA-states',  # USA states mode
                    color='Count_of_accident',  # Use accident count for coloring
                    hover_name='State Name',  # Show state names on hover
                    title='State-Wise Incident Report',
                    color_continuous_scale='sunset',
                    scope='usa')
#%%
# Show the map
fig.show()

#%%
# SMART Q3: How can we predict the location of crossing warning sign present during railroad accidents, based on the historical data.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn import set_config
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

#%%

#%%

accs2 = accs

#%%

# Independent variables
independent_vars = [
    'Equipment Type', 
    'Incident Number', 
    'View Obstruction', 
    'Driver In Vehicle', 
    'Track Type', 
    'Incident Year', 
    'Visibility'
]

# Target variable
target_var = 'Crossing Warning Location'

#%%
# Load your dataset (ensure accs2 is defined)
# accs2 = pd.read_csv("your_data.csv")  # Uncomment and update with the correct file path

# Encode the target variable if it's categorical (multiclass)
label_encoder = LabelEncoder()
accs2[target_var] = label_encoder.fit_transform(accs2[target_var])

#%%
# Handle missing values (optional: based on your dataset)
accs2 = accs2.dropna()

#%%
from imblearn.over_sampling import SMOTE
# Split features and target
X = accs2[independent_vars]
y = accs2[target_var]

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize SMOTE (to oversample the minority classes)
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Resample the training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#%%
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

#%%
# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
from sklearn.metrics import classification_report, precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
print("Precision for each class:", precision)
print("Recall for each class:", recall)
print("F1-Score for each class:", f1)

#%%
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print("Weighted Precision:", precision_avg)
print("Weighted Recall:", recall_avg)
print("Weighted F1-Score:", f1_avg)
#%%
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42, stratify=y)
y_score = rf_model.predict_proba(X_test)
from itertools import cycle
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#%%
# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
         label=f"Micro-average ROC curve (AUC = {roc_auc['micro']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

#%%

# %%
