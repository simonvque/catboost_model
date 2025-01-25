import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from catboost import CatBoostClassifier

np.random.seed(42)

n_samples = 1500

temperature = np.random.uniform(20, 40, n_samples)
precipitation = np.random.uniform(30, 90, n_samples)
humidity = np.random.uniform(0, 100, n_samples)
signal_strength = np.random.uniform(-100, -30, n_samples)
packet_loss = np.random.uniform(0, 20, n_samples)

noise = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

outage = (
    (temperature > 38).astype(int) |
    (precipitation > 100).astype(int) |
    (signal_strength < -90).astype(int) |
    (packet_loss > 10).astype(int) |
    (humidity > 90).astype(int) |
    noise
)

data = pd.DataFrame({
    'Temperature': temperature,
    'Precipitation': precipitation,
    'Humidity': humidity,
    'Signal Strength': signal_strength,
    'Packet Loss': packet_loss,
    'Outage': outage,
})

print(data.head())

target_col = 'Outage'

X = data.loc[:, data.columns != target_col]
y = data.loc[:, target_col].replace({0: 'No', 1: 'Yes'}).astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training set size: {X_train.shape[0]} samples')
print(f'Testing set size: {X_test.shape[0]} samples')

X_train.head()

y_train.head()

features = list(X_train.columns)

print(y_train.value_counts())
print(y_test.value_counts())

model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, verbose = 100, eval_metric='F1')

model.fit(X_train, y_train, plot=True, eval_set=(X_test, y_test))

y_pred = model.predict(X_test)

print(f'Predictions: {y_pred[:100]}')

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'F1: {f1_score(y_test, y_pred, pos_label="Yes"):.4f}')
print(f'Precision: {precision_score(y_test, y_pred, pos_label="Yes"):.4f}')
print(f'Recall: {recall_score(y_test, y_pred, pos_label="Yes"):.4f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

model.save_model("risk_factor_outage_model.bin")
print("Model saved as risk_factor_outage_model.bin")

