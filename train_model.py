import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
temperature = np.random.uniform(20, 45, n_samples)
humidity = np.random.uniform(30, 100, n_samples)
precipitation = np.random.uniform(0, 150, n_samples)
signal_strength = np.random.uniform(-120, -50, n_samples)
packet_loss = np.random.uniform(0, 50, n_samples)

# Generate logical target (Outage)
outage = (
    (temperature > 40).astype(int) |
    (precipitation > 100).astype(int) |
    (signal_strength < -90).astype(int)
)

# Create DataFrame
data = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Precipitation': precipitation,
    'Signal Strength': signal_strength,
    'Packet Loss': packet_loss,
    'Outage': outage
})

# Split data into features and target
X = data[['Temperature', 'Humidity', 'Precipitation', 'Signal Strength', 'Packet Loss']]
y = data['Outage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Outage', 'Outage'], yticklabels=['No Outage', 'Outage'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save_model("catboost_model.bin")
print("Model saved as catboost_model.bin")
