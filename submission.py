import pandas as pd
import numpy as np
import joblib

model = joblib.load('models/best_model.pkl')

test = pd.read_parquet('data/test.parquet')

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

X_test = generate_features(test)

test_predictions = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'id': test['id'], 'score': test_predictions})
submission.to_csv('submission.csv', index=False)

print('Submission file has been created and saved as submission.csv.')
