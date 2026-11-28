"""Bounded web inputs and JSON results for the shared KNN classifier."""
import numpy as np

from .knn_core import (
    DatasetConfig, KNNModel, ModelConfig, _integer, _number, distance_outline,
    generate_dataset, plot_limits, training_data,
)


def integer(value, name, low, high):
    value = _integer(value, name, low)
    if value > high:
        raise ValueError(f"{name} must be at most {high}.")
    return value


def dataset(body):
    config = DatasetConfig(
        dataset=body.get("dataset", "Blobs"),
        n_samples=integer(body.get("n_samples", 180), "Samples", 20, 1200),
        n_classes=integer(body.get("n_classes", 3), "Classes", 2, 6),
        noise=body.get("noise", 0.65),
        separation=_number(body.get("separation", 3), "Pattern size", 0.2, 8),
        imbalance=body.get("imbalance", 1),
        seed=integer(body.get("seed", 42), "Seed", 0, 999999),
    )
    X, y = generate_dataset(config, body.get("stretch", 1), body.get("rotation", 0))
    return training_data(X, y, np.zeros(len(y), dtype=bool))


def classify(body):
    points = body.get("points")
    if not isinstance(points, list) or not 1 <= len(points) <= 1500:
        raise ValueError("Expected 1–1500 training points.")
    config = ModelConfig(
        k=integer(body.get("k", 5), "k", 1, 100),
        metric=body.get("metric", "euclidean"),
        weights=body.get("weights", "uniform"),
        standardize=body.get("standardize", False),
        p=_number(body.get("p", 2), "Minkowski p", 1, 8),
    )
    if not isinstance(config.standardize, bool):
        raise ValueError("Expected a boolean for standardize.")
    model = KNNModel(config).fit(points, body.get("labels"))
    if np.any((model.classes_ < 0) | (model.classes_ > 5)):
        raise ValueError("Labels must be between 0 and 5.")
    result = dict(k=model.effective_k, predictions=[])
    queries = body.get("queries", [])
    if not isinstance(queries, list) or len(queries) > 100:
        raise ValueError("Expected up to 100 query points.")
    if queries:
        queries = model._points(queries)
        votes = model.predict_proba(queries)
        indices, distances, weights = model.neighbors(queries[-1])
        result.update(predictions=model.classes_[votes.argmax(axis=1)].tolist(),
                      classes=model.classes_.tolist(), shares=votes[-1].tolist(),
                      indices=indices.tolist(), distances=distances.tolist(), weights=weights.tolist(),
                      outline=distance_outline(model, queries[-1], distances[-1]).tolist())
    if body.get("resolution") is not None:
        n = integer(body["resolution"], "Resolution", 40, 200)
        bounds = np.asarray(body.get("bounds", plot_limits(model.X_)), dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Invalid plot bounds.")
        # Sample cell centers; row zero corresponds to the bottom of the plot.
        axes = [lo + (np.arange(n) + 0.5) * (hi - lo) / n for lo, hi in bounds]
        grid = np.stack(np.meshgrid(*axes), axis=-1)
        result["regions"] = model.predict(grid.reshape(-1, 2)).reshape(n, n).tolist()
    return result
