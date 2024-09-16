# reuse_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from feature_engineering import FeatureEngineering

# Load the saved PNN model
pnn = joblib.load('trained_pnn_model.pkl')

# Load the saved scaler for data normalization
scaler = joblib.load('scaler.pkl')

# Load new earthquake data to make predictions
new_data_path = './src/data/new_earthquake_data.csv'
new_df = pd.read_csv(new_data_path, low_memory=False)

# Feature Engineering for new data
feature_engineer = FeatureEngineering(window_size=10)
new_features_df = feature_engineer.extract_features(new_df)

# Prepare the new features
X_new = new_features_df.drop(columns=['target']).values

# Normalize the new features using the same scaler used during training
X_new_scaled = scaler.transform(X_new)

# Predict using the loaded PNN model
y_new_pred = pnn.predict(X_new_scaled)

# Display the predictions
print("Predictions for new data:", y_new_pred)

# If you have actual targets for the new data, you can also evaluate the predictions
def evaluate_model(y_true, y_pred):
    # Plot confusion matrix
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print("\nConfusion Matrix:\n", confusion_matrix)

    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    return confusion_matrix, accuracy

# If you have actual target values for the new data, uncomment the following lines:
# y_new_actual = new_features_df['target'].values
# confusion_matrix, accuracy = evaluate_model(y_new_actual, y_new_pred)

# Plot the confusion matrix if actual values are available
# plt.figure(figsize=(10, 6))
# plt.matshow(confusion_matrix, cmap='Blues', fignum=1)
# plt.colorbar()
# plt.title('Confusion Matrix', pad=20)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
