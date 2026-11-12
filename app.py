"""Run the application with the Waitress production WSGI server."""
from waitress import serve

from lab import create_app

app = create_app()

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=5000)
