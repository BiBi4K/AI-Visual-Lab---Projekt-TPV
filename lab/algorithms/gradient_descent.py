"""Linear regression optimizers adapted from Gradient Descent.ipynb."""
import numpy as np

from .kmeans import integer


def real(value, name, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name}: wymagana liczba od {low} do {high}.")
    return value


class GradientDescentOptimizer:
    def __init__(self, X, y):
        self.X, self.y, self.m = X, y, len(y)

    def _compute_loss(self, theta):
        return float(np.sum((self.X @ theta - self.y) ** 2) / (2 * self.m))

    def run(self, mode, learning_rate, max_iterations, beta, initial_theta, rng):
        theta = initial_theta.copy()
        velocity = np.zeros_like(theta)
        history = []

        def frame(iteration):
            return {"iteration": iteration, "theta": theta.ravel().tolist(),
                    "loss": self._compute_loss(theta)}

        history.append(frame(0))
        diverged = False
        for iteration in range(1, max_iterations + 1):
            # Keep unstable experiments JSON-safe and show their last valid state.
            with np.errstate(over="ignore", invalid="ignore"):
                if mode == "SGD":
                    for i in rng.permutation(self.m):
                        xi, yi = self.X[i:i + 1], self.y[i:i + 1]
                        theta -= learning_rate * xi.T @ (xi @ theta - yi)
                else:
                    gradients = self.X.T @ (self.X @ theta - self.y) / self.m
                    if mode == "Momentum":
                        velocity = beta * velocity + learning_rate * gradients
                        theta -= velocity
                    else:
                        theta -= learning_rate * gradients
                state = frame(iteration)
            if not np.isfinite(state["loss"]) or state["loss"] > 1e12:
                diverged = True
                break
            history.append(state)
        return {"mode": mode, "history": history, "diverged": diverged,
                "learning_rate": learning_rate, "beta": beta}


def gradient_descent(n=40, noise=0.5, seed=50, mode="GD", learning_rate=0.1,
                     max_iterations=120, beta=0.44, compare=False):
    integer(n, "Liczba próbek", 5, 1000)
    integer(seed, "Seed", 0, 2**32 - 1)
    integer(max_iterations, "Limit iteracji", 5, 500)
    real(noise, "Szum", 0, 10)
    real(learning_rate, "Learning rate", 0.0001, 2)
    real(beta, "Beta", 0, 0.99)
    if not isinstance(mode, str) or mode not in ("GD", "SGD", "Momentum"):
        raise ValueError("Wybierz GD, SGD lub Momentum.")
    if not isinstance(compare, bool):
        raise ValueError("Porównanie musi być wartością logiczną.")
    rng = np.random.default_rng(seed)
    x = 2 * rng.random((n, 1))
    y = 4 + 3 * x + noise * rng.standard_normal((n, 1))
    X = np.c_[np.ones((n, 1)), x]
    initial = rng.standard_normal((2, 1))
    optimizer = GradientDescentOptimizer(X, y)
    configs = [(mode, learning_rate, beta)]
    if compare:
        configs += [("GD", 0.1, 0), ("SGD", 0.01, 0), ("Momentum", 0.05, 0.9)]
    runs = [optimizer.run(m, lr, max_iterations, b, initial,
                          np.random.default_rng(seed)) for m, lr, b in configs]
    return {"points": np.c_[x, y].tolist(), "runs": runs, "seed": seed}
