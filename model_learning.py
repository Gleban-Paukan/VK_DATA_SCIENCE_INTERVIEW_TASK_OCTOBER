import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Step 1: Data Analysis (EDA)
train = pd.read_parquet('data/train.parquet')
test = pd.read_parquet('data/test.parquet')

# Quick overview of train data
print(train.head())

# Checking for missing values
print('Missing values in train data:', train.isnull().sum())
print('Missing values in test data:', test.isnull().sum())

# Plotting some sample time series to understand the trends and seasonality
for i in range(5):
    plt.plot(train['dates'][i], train['values'][i], label=f'Series {i}')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Sample Time Series from Training Data')
plt.legend()
plt.show()

# Step 2: Handling Missing Values
# Fill missing values with forward fill or backward fill as appropriate
# train.fillna(method='ffill', inplace=True)
# test.fillna(method='ffill', inplace=True)

# Step 3: Feature Engineering
# Function to generate features from the time series
def generate_features(data):
    features = []
    for _, row in data.iterrows():
        series = np.array(row['values'])
        features.append([
            np.mean(series),
            np.std(series),
            np.max(series),
            np.min(series),
            np.percentile(series, 25),
            np.percentile(series, 75),
            np.median(series),
            np.var(series),
            np.mean(np.diff(series)),
            np.max(np.diff(series)),
        ])
    columns = [
        'mean', 'std', 'max', 'min',
        'percentile_25', 'percentile_75',
        'median', 'variance',
        'mean_diff', 'max_diff'
    ]
    return pd.DataFrame(features, columns=columns)

X_train = generate_features(train)
y_train = train['label']
X_test = generate_features(test)

# Step 4: Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training
# Using RandomForest as a baseline model
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model from GridSearch
best_model = grid_search.best_estimator_
print(f'Best Parameters: {grid_search.best_params_}')

# Step 6: Evaluation on Train Data
train_predictions = best_model.predict_proba(X_train_scaled)[:, 1]
roc_auc = roc_auc_score(y_train, train_predictions)
print(f'Training ROC AUC: {roc_auc}')

# Analyzing Feature Importance
feature_importances = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X_train.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Analysis')
plt.show()

# Step 7: Saving the Model
joblib.dump(best_model, 'models/best_model.pkl')

# Step 8: Prediction on Test Data
# Script to generate submission.csv
model = joblib.load('models/best_model.pkl')

# Predictions
test_predictions = model.predict_proba(X_test_scaled)[:, 1]

# Creating submission file
submission = pd.DataFrame({'id': test['id'], 'score': test_predictions})
submission.to_csv('submission.csv', index=False)

print('Submission file has been created.')
