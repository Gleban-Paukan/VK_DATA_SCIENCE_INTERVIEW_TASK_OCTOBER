import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

train = pd.read_parquet('data/train.parquet')

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

X_train['label'] = y_train
X_train.to_csv('data/train_features.csv', index=False)

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', verbose=2, n_jobs=-1)
grid_search.fit(X_train.drop(columns=['label']), y_train)

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'models/best_model.pkl')

print('Model training is complete and the best model has been saved.')
