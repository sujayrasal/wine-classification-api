import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Wine dataset (3 wine cultivars)
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("✅ Wine dataset loaded!")
print(f"Shape: {X.shape}")
print(f"Classes: {wine.target_names}")
print("\nFirst few rows:")
print(X.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Save model and scaler
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ model.pkl saved!")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ scaler.pkl saved!")

# Verify
loaded_model = pickle.load(open('models/model.pkl', 'rb'))
loaded_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

sample = np.array([[13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]])
print(f"\nTest sample prediction: {wine.target_names[loaded_model.predict(loaded_scaler.transform(sample))[0]]}")
print("✅ Expt 1 Complete!")