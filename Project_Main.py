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


#%%
# Step 5: Correlation Analysis - focusing on "Vehicle damage"
correlation_vehicle_damage = accs.corr()['Vehicle Damage Cost'].sort_values(ascending=False)

# Plotting the correlation for "Vehicle damage"
plt.figure(figsize=(8, 6))
sns.barplot(x=correlation_vehicle_damage.index, y=correlation_vehicle_damage.values, palette='coolwarm')
plt.title('Correlation of Other Variables with Vehicle Damage')
plt.xticks(rotation=45)
plt.ylabel('Correlation Coefficient')
plt.show()

#%%
# Step 5: Correlation Analysis - focusing on "Vehicle damage"
correlation_vehicle_damage = accs.corr()['Vehicle Damage Cost'].sort_values(ascending=False)

# Print numerical values of correlation
print("Correlation with Vehicle Damage:")
print(correlation_vehicle_damage)
# %%
#Q3 Time Series Analysis for forecasting future accidents

accs.info()
# %%
accs['Date'] = pd.to_datetime(accs['Date'])
# %%
yearly_accidents = accs.groupby('Report Year').size()

#%%
df = pd.DataFrame({
    'Year': yearly_accidents.index,
    'Accidents': yearly_accidents.values
})
# %%
# Plot the time series
plt.figure(figsize=(12, 6))
yearly_accidents.plot()
plt.title("Monthly Railroad Accidents")
plt.xlabel("Date")
plt.ylabel("Number of Accidents")
plt.grid(True)
plt.show()

# %%
# Filter out features with correlation below 0.1 or between -0.4 and 0.1
filtered_correlations = correlation_vehicle_damage[(correlation_vehicle_damage >= 0.1) | (correlation_vehicle_damage <= -0.07)]

# Print the filtered correlation values
print("Filtered Correlations with Vehicle Damage:")
print(filtered_correlations)

# %%
cols_to_keep =['Highway User','Report Year','Train Speed','Visibility','Vehicle Damage Cost']
accs2 = accs[cols_to_keep]
#%%
accs2.head()

# %%
# Count the number of zeros in the "Vehicle damage" column
num_zeros = (accs['Vehicle Damage Cost'] == 0).sum()

# Print the result
print(f"Total number of zeros in 'Vehicle damage' column: {num_zeros}")
# %%
# Remove rows where 'Vehicle damage' is 0
accs_filtered = accs[accs['Vehicle Damage Cost'] != 0]

# Check the shape of the filtered dataset to confirm the rows are removed
print(f"Original dataset shape: {accs.shape}")
print(f"Filtered dataset shape: {accs_filtered.shape}")

# Optionally, you can view the first few rows of the filtered dataset
print(accs_filtered.head())
# %%
cols_to_keep =['Highway User','Report Year','Train Speed','Visibility','Vehicle Damage Cost']
accs2 = accs_filtered[cols_to_keep]
# %%
# Find the highest and lowest values in the 'Vehicle damage' column after filtering
max_value = accs_filtered['Vehicle Damage Cost'].max()
min_value = accs_filtered['Vehicle Damage Cost'].min()

# Print the highest and lowest values
print(f"Highest value in 'Vehicle damage' column: {max_value}")
print(f"Lowest value in 'Vehicle damage' column: {min_value}")
# %%
X = accs.drop(columns=['Vehicle Damage Cost'])  # Drop the target column from features
y = accs['Vehicle Damage Cost']
# %%
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# %%
from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Print the model coefficients (if needed)
print(f"Model coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# 3. Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# 4. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)


# Print the results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(rmse)
print(mse)
# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot to detect outliers in 'Vehicle damage' column
plt.figure(figsize=(8, 6))
sns.boxplot(x=accs_filtered['Vehicle Damage Cost'])
plt.title('Boxplot for Vehicle Damage')
plt.show()


# %%
from scipy.stats import zscore

# Calculate Z-scores for the 'Vehicle damage' and 'Cost' columns
vehicle_damage_z = zscore(accs_filtered['Vehicle Damage Cost'])


# Identify outliers where Z-score > 3 or < -3
outliers_vehicle_damage = accs_filtered[vehicle_damage_z > 3]


# Print outliers
print(f"Outliers in 'Vehicle damage':\n{outliers_vehicle_damage}")
# %%
# Calculate the IQR for 'Vehicle damage' and 'Cost'
Q1_vehicle_damage = accs_filtered['Vehicle Damage Cost'].quantile(0.25)
Q3_vehicle_damage = accs_filtered['Vehicle Damage Cost'].quantile(0.75)
IQR_vehicle_damage = Q3_vehicle_damage - Q1_vehicle_damage



# Calculate the lower and upper bounds for both columns
lower_bound_vehicle_damage = Q1_vehicle_damage - 1.5 * IQR_vehicle_damage
upper_bound_vehicle_damage = Q3_vehicle_damage + 1.5 * IQR_vehicle_damage


# Remove outliers using the IQR method
accs_filtered_no_outliers = accs_filtered[
    (accs_filtered['Vehicle Damage Cost'] >= lower_bound_vehicle_damage) & 
    (accs_filtered['Vehicle Damage Cost'] <= upper_bound_vehicle_damage) 
]

# Check the shape of the dataset before a

# %%
X = accs_filtered_no_outliers.drop(columns=['Vehicle Damage Cost'])  # Drop the target column from features
y = accs_filtered_no_outliers['Vehicle Damage Cost']
# %%
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# %%
from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model1 = LinearRegression()

# Train the model using the training data
model1.fit(X_train, y_train)

# Print the model coefficients (if needed)
print(f"Model coefficients: {model1.coef_}")
print(f"Intercept: {model1.intercept_}")

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# Make predictions on the test set
y_pred = model1.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# 3. Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# 4. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# %%
# Print the results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(rmse)
print(mse)
# %%
from sklearn.linear_model import RidgeCV
# Initialize and fit Ridge Regression with cross-validation
ridge_model = RidgeCV(alphas=[0.1, 1, 10, 100], store_cv_values=True)
ridge_model.fit(X_train, y_train)
# %%
# Make predictions
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the model
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
# %%
print(f"Ridge Regression R²: {r2_ridge:.4f}")
print(f"Ridge Regression MAE: {mae_ridge:.4f}")
# %%
from sklearn.linear_model import LassoCV

# Initialize and fit Lasso Regression with cross-validation
lasso_model = LassoCV(cv=5)
lasso_model.fit(X_train, y_train)

# Make predictions
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the model
r2_lasso = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print(f"Lasso Regression R²: {r2_lasso:.4f}")
print(f"Lasso Regression MAE: {mae_lasso:.4f}")

# %%
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV

lasso_model = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5)

# Fit the model
lasso_model.fit(X_train, y_train)

# Make predictions
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the model
r2_lasso = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

    # %%
# Print performance metrics
print(f"Lasso Regression R²: {r2_lasso:.4f}")
print(f"Lasso Regression MAE: {mae_lasso:.4f}")
# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
print(f"Decision Tree R²: {r2_score(y_test, y_pred_dt):.4f}")
print(f"Decision Tree MAE: {mean_absolute_error(y_test, y_pred_dt):.4f}")

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Best parameters and model
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best MAE: {-grid_search.best_score_}")

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize the Decision Tree model with pruning
dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)

# Print the performance metrics
print(f"Pruned Decision Tree R²: {r2_dt:.4f}")
print(f"Pruned Decision Tree MAE: {mae_dt:.4f}")

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Print the results
print(f"Random Forest R²: {r2_rf:.4f}")
print(f"Random Forest MAE: {mae_rf:.4f}")

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best MAE: {-grid_search.best_score_}")

# %%


# %%
from sklearn.model_selection import cross_val_score

rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE: {-cv_scores.mean():.4f}")

#%%
