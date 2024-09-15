# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.PNN.feature_engineering import FeatureEngineering
from src.PNN.pnn_model import ProbabilisticNeuralNetwork

# Load the data
data_path = './data/earthquake_data.csv'
df = pd.read_csv(data_path)

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

# Initialize and train the PNN
pnn = ProbabilisticNeuralNetwork(sigma=1.0)
pnn.fit(X_train, y_train)

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
plt.savefig('./visuals/evaluation_plot.png')
plt.show()
