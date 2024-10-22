import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

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

X = generate_features(train)
y = train['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = joblib.load('models/best_model.pkl')

y_val_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation ROC AUC: {auc:.4f}')

y_val_pred_labels = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred_labels)
print(f'Validation Accuracy: {accuracy:.4f}')

cm = confusion_matrix(y_val, y_val_pred_labels)
print('Confusion Matrix:')
print(cm)

fpr, tpr, _ = roc_curve(y_val, y_val_pred)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
