class KNNRegressor:
    def __init__(self, k=3, distance_func=None):
        """Initialize the KNN regressor."""
        if distance_func is None:
            raise ValueError("A distance function must be provided.")
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if not callable(distance_func):
            raise TypeError("distance_func must be a callable function.")

        self.k = k
        self.distance_func = distance_func
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fit the KNN regressor to the training data."""
        self.X_train = X_train
        self.y_train = y_train

    def _get_neighbors(self, x_test):
        """Get the k nearest neighbors of a test point."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model must be fitted before prediction.")

        distances = []

        for train_point, target_value in zip(self.X_train, self.y_train):
            dist = self.distance_func(x_test, train_point)
            distances.append((dist, target_value))

        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

    def _predict_one(self, x_test):
        """Predict the target value of a single test point."""
        neighbors = self._get_neighbors(x_test)
        neighbor_values = [value for _, value in neighbors]
        return sum(neighbor_values) / len(neighbor_values)

    def predict(self, X_test):
        """Predict the target values of multiple test points."""
        predictions = []

        for x in X_test:
            prediction = self._predict_one(x)
            predictions.append(prediction)

        return predictions