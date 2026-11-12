"""Shared dataset generator for all three algorithm modules."""
import numpy as np

from .algorithms.kmeans import integer


def generate(n=200, clusters=3, noise=5, seed=42):
    integer(n, "Liczba punktów", 1, 2000)
    integer(clusters, "Liczba skupisk", 1, 12)
    integer(seed, "Seed", 0, 2**32 - 1)
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not 0 <= noise <= 30:
        raise ValueError("Szum musi należeć do [0, 30].")
    rng = np.random.default_rng(seed)
    centers = rng.uniform(15, 85, (clusters, 2))
    labels = np.arange(n) % clusters
    points = np.clip(centers[labels] + rng.normal(0, noise, (n, 2)), 0, 100)
    return {"points": points.tolist(), "seed": seed}
