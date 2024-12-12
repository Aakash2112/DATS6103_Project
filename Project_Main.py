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
accs1 = accs
#%%
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
# SMART Q: Predicting the Driver Injury

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
#MODEL 1: Random Forest Classifier
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
threshold = 0.01
low_importance_features = importance_df[importance_df['Importance'] < threshold]['Feature']

# Drop the low-importance features from the dataframe
X_reduced = X.drop(columns=low_importance_features)

print("Dropped features with low importance:", low_importance_features)
# %%
# Re-train the model with reduced features
X_train, X_test, y_train1, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
model_reduced = RandomForestClassifier(random_state=42)
model_reduced.fit(X_train, y_train1)

# Evaluate the model on the test set
accuracy = model_reduced.score(X_test, y_test)
print(f"Model accuracy after removing low-importance features: {accuracy:.4f}")

# %%
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
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to display most important features at the top
plt.show()
# %%
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
# Compute the confusion matrix
cm1 = confusion_matrix(y_test, y_pred)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm1, annot=True, fmt='d', cmap='viridis', xticklabels=['Killed', 'Injured', 'Uninjured'], yticklabels=['Killed', 'Injured', 'Uninjured'], cbar=False)

# Add labels and title
plt.title('Confusion Matrix for Random Forest', fontsize=14)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
# %%

# MODEL 2: DECISION TREE
# Initialize the SVM classifier with probability estimates
#X = accs1.drop('Driver Condition', axis=1)  # Features: exclude the target column
#y = accs1['Driver Condition'] 
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_reduced, y, test_size=0.2, random_state=42)



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
plot_tree(best_model, filled=True, feature_names=X_reduced.columns, class_names=[str(cls) for cls in y.unique()], rounded=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

# %%
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
    best_model, X_reduced, y, cv=5, n_jobs=-1
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
# %%

# %%
