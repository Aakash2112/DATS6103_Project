#%%


#%% Import Libraries
import numpy as np
#%%
import pandas as pd
import pandas as pd
print(pd.__version__)
#%%
#%% Load Data
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
# %%


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
categorical_columns = accs2.select_dtypes(include=['object']).columns

# Encode all categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    accs2[col] = le.fit_transform(accs2[col])
    label_encoders[col] = le


accs2.columns
accs2.head()



#%%
# start with first questions. 

# %%
from sklearn.model_selection import train_test_split

accs2.columns
freq = accs2.groupby(['State Name', 'City Name'])['Report Year'].sum()



#%%
# Kmean - clustering 

#%% Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import copy

#%% Load Data
accs = pd.read_csv("accidents_final_data.csv")

#%% Data Inspection
accs.info()  
accs.describe()  

# Check for missing values
missing_values = accs.isnull().sum()
print(f"Missing values:\n{missing_values}")

# Check for duplicates
duplicates = accs.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

# Remove duplicates
accs = accs.drop_duplicates()
print(f"Shape after removing duplicates: {accs.shape}")

#%% Inspect Columns and Data Types
accs.info()

# View unique values in each column
for column in accs.columns:
    print(f"Column: {column}")
    print(accs[column].unique())
    print("=" * 50)

# Create a copy of the dataset for modifications
accs2 = copy.deepcopy(accs)

#%% Encode Categorical Variables
encoder = LabelEncoder()
categorical_columns = accs2.select_dtypes(include=['object']).columns

# Encode all categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    accs2[col] = le.fit_transform(accs2[col])
    label_encoders[col] = le

# Verify encoding
print("Encoded DataFrame:")
print(accs2.head())

accs2
#%% Group by State and City to create 'freq'
freq = accs2.groupby(['State Name', 'City Name'])['Report Year'].sum().reset_index()
freq.rename(columns={'Report Year': 'Incident Frequency'}, inplace=True)
print(freq.head())

freq
#%% Scaling the Frequency for K-Means
scaler = StandardScaler()
freq['Incident Frequency Scaled'] = scaler.fit_transform(freq[['Incident Frequency']])

# Use the scaled frequency as X
X = freq[['Incident Frequency Scaled']]
X
#%% K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
freq['Cluster'] = kmeans.fit_predict(X)

# Add cluster labels to the DataFrame
print("Clustered Data:")
print(freq.head())

#%% Evaluate the Clustering Model
silhouette_avg = silhouette_score(X, freq['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

#%% Interpretation
# Decode the 'State Name' and 'City Name' if needed for human-readable results
freq['State Name Decoded'] = freq['State Name'].apply(lambda x: label_encoders['State Name'].inverse_transform([x])[0])
freq['City Name Decoded'] = freq['City Name'].apply(lambda x: label_encoders['City Name'].inverse_transform([x])[0])

print("Clustered Data with Decoded Names:")
print(freq[['State Name Decoded', 'City Name Decoded', 'Incident Frequency', 'Cluster']].head())


#%%
# 2nd part 

import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier  # For KNN
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.metrics import accuracy_score, classification_report 

#%% Load the dataset
accs = pd.read_csv("accidents_final_data.csv")

#%% Data Cleaning
# Create a copy of the dataset for modifications
accs2 = copy.deepcopy(accs)

# Encode categorical variables
encoder = LabelEncoder()
categorical_columns = accs2.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    accs2[col] = le.fit_transform(accs2[col])
    label_encoders[col] = le

print("Encoded DataFrame:")
print(accs2.head())

#%% Group by State and City to create 'freq'
freq = accs2.groupby(['State Name', 'City Name'])['Report Year'].sum().reset_index()
freq.rename(columns={'Report Year': 'Incident Frequency'}, inplace=True)
print(freq.head())

#%% Scaling the Frequency for K-Means
scaler = StandardScaler()
freq['Incident Frequency Scaled'] = scaler.fit_transform(freq[['Incident Frequency']])

# Use the scaled frequency as X
X = freq[['Incident Frequency Scaled']]

#%% K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
freq['Cluster'] = kmeans.fit_predict(X)

# Add cluster labels to the DataFrame
print("Clustered Data:")
print(freq.head())

#%% Evaluate the Clustering Model
silhouette_avg = silhouette_score(X, freq['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

#%% Interpretation
# Decode the 'State Name' and 'City Name' back to their original labels
freq['State Name Decoded'] = freq['State Name'].apply(lambda x: label_encoders['State Name'].inverse_transform([x])[0])
freq['City Name Decoded'] = freq['City Name'].apply(lambda x: label_encoders['City Name'].inverse_transform([x])[0])

print("Clustered Data with Decoded Names:")
print(freq[['State Name Decoded', 'City Name Decoded', 'Incident Frequency', 'Cluster']].head())

#%% Save the results
#freq.to_csv("clustered_results.csv", index=False)

# %%
###############################################
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

#%% Interpretation
from sklearn.neighbors import KNeighborsClassifier  # For KNN
#%%

# Add the predicted labels to the DataFrame
freq['KNN Cluster Prediction'] = knn.predict(X)

print("Data with KNN Predictions:")
print(freq[['State Name Decoded', 'City Name Decoded', 'Incident Frequency', 'KNN Cluster Prediction']].head())

# %%
#resukt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef,
    log_loss, cohen_kappa_score, balanced_accuracy_score
)
import numpy as np
import matplotlib.pyplot as plt

# Example: y_true (actual values), y_pred (predicted values)
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0])

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

# Recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")

# F1-Score
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# ROC-AUC
roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC-AUC: {roc_auc}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_true, y_pred)
print(f"Matthews Correlation Coefficient: {mcc}")

# Log Loss
# For Log Loss, probabilities are needed (not just binary predictions).
# Here, we assume some predicted probabilities for illustration.
y_pred_proba = np.array([0.1, 0.9, 0.3, 0.6, 0.7, 0.8, 0.9, 0.2, 0.4, 0.3])
log_loss_value = log_loss(y_true, y_pred_proba)
print(f"Log Loss: {log_loss_value}")

# Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa}")

# Balanced Accuracy
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy: {balanced_accuracy}")

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Example: Generate synthetic data for clustering
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='x', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Evaluate clustering performance using different metrics
# Silhouette Score
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette}")

# Adjusted Rand Index (ARI) - compares the similarity between true labels and predicted clusters
# Note: ARI is typically used when true labels are available (in this case 'y' is the true label)
ari = adjusted_rand_score(y, labels)
print(f"Adjusted Rand Index: {ari}")

# Calinski-Harabasz Index - measures the ratio of the sum of between-cluster dispersion to within-cluster dispersion
calinski_harabasz = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz}")

# %%


# feaguring out the clustering centriod 

print(f"Cluster Centers: {centers}")
import numpy as np
unique, counts = np.unique(labels, return_counts=True)
print(f"Cluster Distribution: {dict(zip(unique, counts))}")
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='x', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
silhouette = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette}")

calinski_harabasz = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz}")

ari = adjusted_rand_score(y, labels)  # Only if you have true labels for comparison
print(f"Adjusted Rand Index: {ari}")

# %%

#%%
# cLUSTER WITTH ACTUCAL DATA 
import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace X with your actual data)
freq = np.random.randn(300, 2)  # Replace with your actual dataset
labels  = np.random.choice([0, 1, 2, 3], size=300)  # Replace with your actual labels

# Cluster Centers
cluster_centers = np.array([[-2.70981136, 8.97143336],
                            [-6.83235205, -6.83045748],
                            [4.7182049, 2.04179676],
                            [-8.87357218, 7.17458342]])

# Plot the data points with cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)

# Plot the cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='X', s=200, label='Cluster Centers')

# Add title and labels
plt.title('Cluster Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Cluster Distribution
unique, counts = np.unique(labels, return_counts=True)
cluster_distribution = dict(zip(unique, counts))
print(f"Cluster Distribution: {cluster_distribution}")

# Cluster Centers
print(f"Cluster Centers (Centroids):\n{cluster_centers}")

# %%
