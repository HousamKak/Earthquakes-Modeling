# pnn_model.py

import numpy as np

class ProbabilisticNeuralNetwork:
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.classes = None
        self.training_data = None
        self.training_labels = None

    def fit(self, X, y):
        self.training_data = X
        self.training_labels = y
        self.classes = np.unique(y)

    def _gaussian_kernel(self, x, t):
        distance = np.linalg.norm(x - t)
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-distance ** 2 / (2 * self.sigma ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            class_likelihoods = []
            for cls in self.classes:
                class_data = self.training_data[self.training_labels == cls]
                kernel_sum = np.sum([self._gaussian_kernel(x, t) for t in class_data])
                likelihood = kernel_sum / len(class_data)
                class_likelihoods.append(likelihood)
            predicted_class = self.classes[np.argmax(class_likelihoods)]
            predictions.append(predicted_class)
        return np.array(predictions)
