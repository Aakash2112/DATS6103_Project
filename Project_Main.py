#%%
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
# %%
# Load the data
accs = pd.read_csv("accidents_final_data.csv")
# %%
accs.head()
#%%
# SMART Q: Predicting the Driver Injury
selected_columns = ['Report Year','Month','County Name','State Name','City Name','Highway Name','Highway User','Vehicle Direction','Highway User Position','Temperature','Visibility','Weather Condition','Train Direction','Highway User Action','Driver Condition']

# Subset the DataFrame to only include those columns

accs1 = accs[selected_columns]

#%%
#accs1['Time'] = accs1['Time'].replace(r'(^0:)', '12:', regex=True)
#%%
accs1['Time'] = pd.to_datetime(accs1['Time'], format='%I:%M %p')
accs1['Hour'] = accs1['Time'].dt.hour
accs1['Minute'] = accs1['Time'].dt.minute
#accs['AM_PM'] = (accs['Time'].dt.hour < 12).astype(int)
#%%
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
accs1['County Name'] = encoder.fit_transform(accs1['County Name'])
accs1['State Name'] = encoder.fit_transform(accs1['State Name'])
accs1['City Name'] = encoder.fit_transform(accs1['City Name'])
accs1['Highway Name'] = encoder.fit_transform(accs1['Highway Name'])
accs1['Vehicle Direction'] = encoder.fit_transform(accs1['Vehicle Direction'])
accs1['Highway User Position'] = encoder.fit_transform(accs1['Highway User Position'])
accs1['Visibility'] = encoder.fit_transform(accs1['Visibility'])
accs1['Weather Condition'] = encoder.fit_transform(accs1['Weather Condition'])
accs1['Train Direction'] = encoder.fit_transform(accs1['Train Direction'])
accs1['Highway User Action'] = encoder.fit_transform(accs1['Highway User Action'])
accs1['Driver Condition'] = encoder.fit_transform(accs1['Driver Condition'])
accs1['Highway User'] = encoder.fit_transform(accs1['Highway User'])
# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# %%
X = accs1.drop('Driver Condition', axis=1)  # Features: exclude the target column
y = accs1['Driver Condition'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)  # Use X_train_scaled if scaling is done

#%%
# Make predictions on the test data
y_pred = rf_model.predict(X_test)  # Use X_test_scaled if scaling is done

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

# %%
accs1.head()
# %%
#accs1['Highway User'] = encoder.fit_transform(accs1['Highway User'])
accs1.dtypes
# %%
