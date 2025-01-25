import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


# Step 1: Generate Synthetic Data
np.random.seed(42)
n_samples = 1500

# Generate features
temperature = np.random.uniform(20, 45, n_samples)
humidity = np.random.uniform(30, 100, n_samples)
precipitation = np.random.uniform(0, 150, n_samples)
signal_strength = np.random.uniform(-120, -50, n_samples)
packet_loss = np.random.uniform(0, 50, n_samples)

# Add noise (10% random flipping of target labels)
noise = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# Generate logical target (Outage)
outage = (
    (temperature > 38).astype(int) |
    (precipitation > 100).astype(int) |
    (signal_strength < -90).astype(int) |
    (packet_loss > 10).astype(int) |
    (humidity > 90).astype(int) |
    noise
)

# Create a DataFrame
data = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Precipitation': precipitation,
    'Signal Strength': signal_strength,
    'Packet Loss': packet_loss,
    'Outage': outage
})

print("Dataset Sample:")
print(data.head())


# Step 2: Handle Class Imbalance using SMOTE
X = data[['Temperature', 'Humidity', 'Precipitation', 'Signal Strength', 'Packet Loss']]
y = data['Outage']

# Oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())


# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Step 4: Train CatBoost Model with Hyperparameter Tuning
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    class_weights={0: 2.0, 1: 1.0},  # Balance class weights
    verbose=100
)
model.fit(X_train, y_train)


# Step 5: Evaluate the Model
# Predictions
y_pred = model.predict(X_test)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Outage', 'Outage'], yticklabels=['No Outage', 'Outage'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-Validation
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.2f}")


# Step 6: Analyze Feature Importance
feature_importances = model.get_feature_importance()
plt.barh(X.columns, feature_importances, color='teal')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


# Step 7: Test Custom Scenarios
custom_data = pd.DataFrame({
    'Temperature': [25, 42, 30, 40],
    'Humidity': [60, 95, 50, 91],
    'Precipitation': [5, 120, 10, 105],
    'Signal Strength': [-70, -100, -80, -95],
    'Packet Loss': [5, 15, 0, 12]
})

# Predict custom test cases
custom_predictions = model.predict(custom_data)
print("\nCustom Test Predictions:")
print(custom_data)
print("Predicted Outages:", custom_predictions)


# # Step 8: Save the Model

# # Save the model
# model.save_model("catboost_model_improved.bin")
# print("\nModel saved as 'catboost_model_improved.bin'")

# # Load the model
# loaded_model = CatBoostClassifier()
# loaded_model.load_model("catboost_model_improved.bin")
# print("Model loaded successfully!")
