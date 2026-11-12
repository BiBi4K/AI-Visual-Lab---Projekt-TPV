"""Application factory. Register teammates' blueprints here."""
from .algorithms.kmeans import kmeans
from .datasets import generate


def create_app():
    from flask import Flask, jsonify, render_template, request
    from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024

    @app.get("/")
    def index():
        return render_template("index.html")

    def payload():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError("Oczekiwano obiektu JSON.")
        return body

    @app.post("/api/datasets/generate")
    def dataset():
        body = payload()
        return jsonify(generate(**{key: body[key] for key in ("n", "clusters", "noise", "seed") if key in body}))

    @app.post("/api/kmeans")
    def run_kmeans():
        body = payload()
        return jsonify(kmeans(body.get("points"), body.get("k", 3),
                              body.get("seed", 42), body.get("max_iterations", 100)))

    @app.errorhandler(ValueError)
    @app.errorhandler(BadRequest)
    def invalid_input(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="Zbyt duży zbiór danych (limit 512 KB)."), 413

    return app
