import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from feature_engineering import FeatureEngineering
from pnn_model import ProbabilisticNeuralNetwork

# Load the data
data_path = './src/data/earthquake_data.csv'

# Use low_memory=False to avoid dtype warning from large CSV file
df = pd.read_csv(data_path, low_memory=False)

# Feature Engineering
feature_engineer = FeatureEngineering(window_size=10)
features_df = feature_engineer.extract_features(df)

# Split data into features and target
X = features_df.drop(columns=['target']).values
y = features_df['target'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check if the PNN model file exists
model_filename = 'trained_pnn_model.pkl'

if os.path.exists(model_filename):
    # Delete the existing model
    print(f"Model '{model_filename}' found. Deleting the existing model.")
    os.remove(model_filename)
    print(f"Deleted '{model_filename}'. Retraining the model.")
else:
    print(f"Model '{model_filename}' not found. Training a new model.")

# Initialize and train the PNN
pnn = ProbabilisticNeuralNetwork(sigma=1.0)
pnn.fit(X_train, y_train)

# Save the trained PNN model to a file
joblib.dump(pnn, model_filename)
print(f"New model saved as '{model_filename}'.")

# Predict on the test set
y_pred = pnn.predict(X_test)

# Evaluate the model
def evaluate_model(y_test, y_pred):
    # Plot confusion matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print("\nConfusion Matrix:\n", confusion_matrix)

    # Accuracy
    accuracy = np.mean(y_test == y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    return confusion_matrix, accuracy

confusion_matrix, accuracy = evaluate_model(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 6))
plt.matshow(confusion_matrix, cmap='Blues', fignum=1)
plt.colorbar()
plt.title('Confusion Matrix', pad=20)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
