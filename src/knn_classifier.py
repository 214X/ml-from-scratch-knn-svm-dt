from collections import Counter
import numpy as np


class KNNClassifier:
    def __init__(self, k=3, distance_func=None):
        """Initialize the KNN classifier."""
        # Validate the input parameters
        if distance_func is None:
            raise ValueError("A distance function must be provided.")
        # Validate that the distance function is callable
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        # Validate that the distance function is callable
        if not callable(distance_func):
            raise TypeError("distance_func must be a callable function.")
        
        self.k = k
        self.distance_func = distance_func
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fit the KNN classifier to the training data."""
        self.X_train = X_train
        self.y_train = y_train

    def _get_neighbors(self, x_test):
        """Get the k nearest neighbors of a test point."""

        # Ensure that the model has been fitted before trying to get neighbors
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model must be fitted before prediction.")

        distances = []

        for train_point, label in zip(self.X_train, self.y_train):
            dist = self.distance_func(x_test, train_point)
            distances.append((dist, label))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        return neighbors

    def _predict_one(self, x_test):
        """Predict the label of a single test point."""
        neighbors = self._get_neighbors(x_test)

        neighbor_labels = [label for _, label in neighbors]

        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        return most_common_label

    def predict(self, X_test):
        """Predict the labels of multiple test points."""
        predictions = []

        for x in X_test:
            prediction = self._predict_one(x)
            predictions.append(prediction)

        return predictions

    def predict_proba(self, X_test):
        """Estimate class probabilities from neighbor label frequencies."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model must be fitted before prediction.")

        classes = np.unique(self.y_train)
        probabilities = []

        for x in X_test:
            neighbors = self._get_neighbors(x)
            neighbor_labels = [label for _, label in neighbors]
            label_counts = Counter(neighbor_labels)
            probabilities.append([
                label_counts.get(class_label, 0) / self.k
                for class_label in classes
            ])

        return np.array(probabilities)
