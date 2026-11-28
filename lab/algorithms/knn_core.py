"""Small, inspectable K-nearest-neighbours classifier and 2D toy datasets.

The classifier deliberately implements distance calculation and voting in NumPy
so the visualization can show exactly which training points cast each vote.
scikit-learn is used only to generate the familiar blobs, moons and circles.
"""

from dataclasses import dataclass, replace

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons


DATASETS = ("Blobs", "Moons", "Circles", "Spirals", "XOR")
METRICS = ("euclidean", "manhattan", "chebyshev", "minkowski")


@dataclass
class DatasetConfig:
    dataset: str = "Blobs"
    n_samples: int = 180
    n_classes: int = 3
    noise: float = 0.65
    separation: float = 3.0
    imbalance: float = 1.0
    seed: int = 42


@dataclass
class ModelConfig:
    k: int = 5
    metric: str = "euclidean"
    p: float = 2.0
    weights: str = "uniform"
    standardize: bool = False


def _integer(value, name, minimum=0):
    """Reject fractional or non-finite integer controls with a useful message."""
    try:
        number = float(value)
        if not np.isfinite(number) or number != int(number) or number < minimum:
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer of at least {minimum}.") from None


def _number(value, name, minimum, maximum=None):
    try:
        value = float(value)
        if not np.isfinite(value) or value < minimum:
            raise ValueError
        if maximum is not None and value > maximum:
            raise ValueError
        return value
    except (TypeError, ValueError, OverflowError):
        interval = f"{minimum}–{maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{name} must be a finite number in {interval}.") from None


def _class_counts(total, classes, imbalance):
    """Round target class proportions, preserving the total and every class."""
    weights = np.ones(classes)
    weights[-1] = imbalance
    targets = total * weights / weights.sum()
    counts = np.maximum(1, np.floor(targets).astype(int))
    while counts.sum() < total:
        counts[np.argmax(targets - counts)] += 1
    while counts.sum() > total:
        removable = np.where(counts > 1, counts - targets, -np.inf)
        counts[np.argmax(removable)] -= 1
    return counts


def _generate_pattern(config: DatasetConfig):
    """Return reproducible ``X: (n, 2)`` and integer class labels ``y: (n,)``.

    ``noise`` is the standard deviation of Gaussian coordinate noise in plot
    units. ``separation`` scales the underlying geometry before that noise is
    added. ``imbalance`` is the desired population of the last class relative
    to each other class; exact proportions are rounded to preserve sample count.
    Moons, circles and XOR always have two classes.
    """
    if config.dataset not in DATASETS:
        raise ValueError(f"Choose a dataset from {', '.join(DATASETS)}.")
    classes = 2 if config.dataset in ("Moons", "Circles", "XOR") else _integer(
        config.n_classes, "Class count", 2
    )
    total = _integer(config.n_samples, "Sample count", classes)
    noise = _number(config.noise, "Noise", 0, 2)
    separation = _number(config.separation, "Separation", 0)
    if separation == 0:
        raise ValueError("Separation must be greater than zero.")
    imbalance = _number(config.imbalance, "Class balance", 0.1, 1)
    seed = _integer(config.seed, "Random seed", 0)
    if seed > 2**32 - 1:
        raise ValueError("Random seed must be at most 4294967295.")
    counts = _class_counts(total, classes, imbalance)
    rng = np.random.default_rng(seed)

    if config.dataset == "Blobs":
        angles = np.linspace(0, 2 * np.pi, classes, endpoint=False) + np.pi / 2
        centers = separation * np.column_stack((np.cos(angles), np.sin(angles)))
        X, y = make_blobs(n_samples=counts.tolist(), centers=centers,
                          cluster_std=noise, random_state=seed)
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

    if config.dataset == "Moons":
        X, y = make_moons(n_samples=tuple(counts), noise=None, random_state=seed)
        X = (X - np.array([0.5, 0.25])) * separation
    elif config.dataset == "Circles":
        X, y = make_circles(n_samples=tuple(counts), noise=None, factor=0.45,
                            random_state=seed)
        X = X * separation
    elif config.dataset == "Spirals":
        pieces = []
        for label, count in enumerate(counts):
            position = rng.uniform(0.06, 1, count)
            angle = 2.5 * np.pi * position + label * 2 * np.pi / classes
            radius = separation * position
            pieces.append(np.column_stack((radius * np.cos(angle), radius * np.sin(angle))))
        X = np.concatenate(pieces)
        y = np.repeat(np.arange(classes), counts)
    else:  # XOR: opposite quadrants share a class.
        pieces = []
        for label, count in enumerate(counts):
            magnitudes = rng.uniform(0.15, 1, (count, 2)) * separation
            signs = rng.choice([-1, 1], count)
            magnitudes[:, 0] *= signs
            magnitudes[:, 1] *= signs * (1 if label == 0 else -1)
            pieces.append(magnitudes)
        X = np.concatenate(pieces)
        y = np.repeat(np.arange(classes), counts)

    X = X + rng.normal(0, noise, X.shape)
    order = rng.permutation(total)
    return X[order], y[order].astype(int)


def generate_dataset(config: DatasetConfig, stretch=1, rotation=0):
    X, y = _generate_pattern(config)
    X[:, 0] *= _number(stretch, "Stretch", 0.1, 10)
    angle = np.deg2rad(_number(rotation, "Rotation", 0, 180))
    transform = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return X @ transform, y


def training_data(X, y, added):
    return dict(version=1, points=X.tolist(), labels=y.tolist(), added_by_click=added.tolist())


def plot_limits(points):
    """Return a square view with space to click outside the training clusters."""
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * 0.64, 1.0)
    return (center[0] - radius, center[0] + radius), (center[1] - radius, center[1] + radius)


def distance_outline(model, point, radius):
    """Map the selected distance's radius back into the original feature units."""
    theta = np.linspace(0, 2 * np.pi, 401)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    metric = model.config.metric
    if metric == "chebyshev":
        norm = np.abs(directions).max(axis=1)
    else:
        p = {"euclidean": 2, "manhattan": 1}.get(metric, model.config.p)
        norm = (np.abs(directions) ** p).sum(axis=1) ** (1 / p)
    return point + directions / norm[:, None] * radius * model.scale_


class KNNModel:
    """KNN classification with transparent distance calculation and voting.

    ``fit`` stores the examples. For a query, distances select the closest k
    examples and their labels vote. Equal class vote totals select the lowest
    label. Equal distances select earlier training rows first. Reported class
    probabilities are normalized vote shares, not calibrated probabilities.
    """

    def __init__(self, config: ModelConfig):
        self.config = replace(config)
        self.effective_k = 0

    def fit(self, X, y):
        X = self._points(X)
        if len(X) == 0:
            raise ValueError("Add at least one training point before fitting KNN.")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Labels must be a one-dimensional array matching the points.")
        try:
            numeric_y = y.astype(float)
        except (TypeError, ValueError):
            raise ValueError("Class labels must be finite integers.") from None
        if not np.isfinite(numeric_y).all() or np.any(numeric_y != np.floor(numeric_y)):
            raise ValueError("Class labels must be finite integers.")
        self.config.k = _integer(self.config.k, "k", 1)
        if self.config.metric not in METRICS:
            raise ValueError(f"Choose a distance metric from {', '.join(METRICS)}.")
        if self.config.weights not in ("uniform", "distance"):
            raise ValueError("Voting weights must be 'uniform' or 'distance'.")
        self.config.p = _number(self.config.p, "Minkowski p", 1)
        self.X_ = X.copy()
        self.y_ = numeric_y.astype(int)
        self.classes_, self._class_indices = np.unique(self.y_, return_inverse=True)
        self.effective_k = min(self.config.k, len(X))
        self.mean_ = X.mean(axis=0) if self.config.standardize else np.zeros(2)
        self.scale_ = X.std(axis=0) if self.config.standardize else np.ones(2)
        self.scale_[self.scale_ == 0] = 1.0
        self._training = (X - self.mean_) / self.scale_
        return self

    @staticmethod
    def _points(points):
        try:
            points = np.asarray(points, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("Points must contain numeric x and y coordinates.") from None
        if points.ndim == 1 and points.shape == (2,):
            points = points.reshape(1, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must have shape (number of points, 2).")
        if not np.isfinite(points).all():
            raise ValueError("Point coordinates must be finite numbers.")
        return points

    def _check_fitted(self):
        if not self.effective_k:
            raise ValueError("Fit the model before making predictions.")

    def _nearest(self, points):
        """Compute distances in transformed space, then choose k neighbours."""
        transformed = (points - self.mean_) / self.scale_
        delta = np.abs(transformed[:, None, :] - self._training[None, :, :])
        if self.config.metric == "euclidean":
            distances = np.sqrt(np.sum(delta * delta, axis=2))
        elif self.config.metric == "manhattan":
            distances = np.sum(delta, axis=2)
        elif self.config.metric == "chebyshev":
            distances = np.max(delta, axis=2)
        else:
            # Scale before exponentiation to avoid overflow for large p.
            maximum = np.max(delta, axis=2)
            safe = np.where(maximum == 0, 1, maximum)
            distances = maximum * np.sum((delta / safe[:, :, None]) ** self.config.p,
                                          axis=2) ** (1 / self.config.p)
        # Stable sorting makes equal-distance neighbourhoods reproducible.
        indices = np.argsort(distances, axis=1, kind="stable")[:, :self.effective_k]
        return indices, np.take_along_axis(distances, indices, axis=1)

    def _weights(self, distances):
        if self.config.weights == "uniform":
            return np.full_like(distances, 1 / self.effective_k)
        zero = distances == 0
        # Scale inverse distances by the closest positive distance; normalization
        # gives the same vote shares as 1/d without overflowing near zero.
        closest = distances[:, :1]
        weights = np.divide(closest, distances, out=np.zeros_like(distances),
                            where=distances > 0)
        weights = np.where(zero.any(axis=1, keepdims=True), zero.astype(float), weights)
        return weights / weights.sum(axis=1, keepdims=True)

    def neighbors(self, point):
        """Return sorted training indices, distances and normalized vote weights.

        If distance voting finds an exact coordinate match, only exact matches
        vote; all other selected neighbours receive weight zero.
        """
        self._check_fitted()
        point = self._points(point)
        if len(point) != 1:
            raise ValueError("Inspect one query point at a time.")
        indices, distances = self._nearest(point)
        return indices[0], distances[0], self._weights(distances)[0]

    def predict_proba(self, points):
        """Return vote shares, with columns ordered as ``classes_``.

        Work in batches so a dense decision-map grid does not allocate a full
        grid-by-training-points distance matrix in memory.
        """
        self._check_fitted()
        points = self._points(points)
        result = np.zeros((len(points), len(self.classes_)))
        batch_size = max(1, min(512, 2_000_000 // len(self._training)))
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            indices, distances = self._nearest(points[start:stop])
            weights = self._weights(distances)
            labels = self._class_indices[indices]
            for class_index in range(len(self.classes_)):
                result[start:stop, class_index] = np.sum(
                    weights * (labels == class_index), axis=1
                )
        return result

    def predict(self, points):
        """Return the winning class for each query (lowest label breaks ties)."""
        votes = self.predict_proba(points)
        return self.classes_[np.argmax(votes, axis=1)]
