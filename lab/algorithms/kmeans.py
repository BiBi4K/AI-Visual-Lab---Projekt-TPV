"""Lloyd's algorithm with reproducible k-means++ initialization."""
import numpy as np


def points_array(points):
    try:
        data = np.asarray(points, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Punkty muszą być parami liczb.") from exc
    if data.ndim != 2 or data.shape[1] != 2 or not 1 <= len(data) <= 2000:
        raise ValueError("Podaj od 1 do 2000 punktów [x, y].")
    if not np.isfinite(data).all() or (np.abs(data) > 1000).any():
        raise ValueError("Współrzędne muszą być skończone i należeć do [-1000, 1000].")
    return data


def integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name}: wymagana liczba całkowita od {low} do {high}.")
    return value


def kmeans(points, k=3, seed=42, max_iterations=100):
    data = points_array(points)
    integer(k, "k", 1, min(12, len(np.unique(data, axis=0))))
    integer(seed, "Seed", 0, 2**32 - 1)
    integer(max_iterations, "Limit iteracji", 1, 200)
    rng = np.random.default_rng(seed)
    centers = [data[rng.integers(len(data))].copy()]
    for _ in range(1, k):
        distances = ((data[:, None] - np.array(centers)) ** 2).sum(axis=2).min(axis=1)
        centers.append(data[rng.choice(len(data), p=distances / distances.sum())].copy())
    centers = np.array(centers)

    def frame(iteration):
        distances = ((data[:, None] - centers) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        return labels, {"iteration": iteration, "centroids": centers.tolist(),
                        "labels": labels.tolist(),
                        "inertia": float(distances[np.arange(len(data)), labels].sum())}

    labels, initial = frame(0)
    history = [initial]
    converged = False
    for iteration in range(1, max_iterations + 1):
        # An empty cluster retains its center, so the objective cannot increase.
        updated = np.array([data[labels == j].mean(axis=0) if (labels == j).any()
                            else centers[j] for j in range(k)])
        converged = bool(np.max(np.linalg.norm(updated - centers, axis=1)) <= 1e-7)
        centers = updated
        labels, state = frame(iteration)
        history.append(state)
        if converged:
            break
    return {"history": history, "converged": converged, "k": k, "seed": seed}
