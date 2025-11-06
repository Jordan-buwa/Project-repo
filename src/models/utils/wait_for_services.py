import time
import socket
import sys
import urllib.request
import os


def exponential_backoff(base_delay, attempt, max_delay=60):
    """Calculate delay with exponential backoff."""
    return min(base_delay * (2 ** attempt), max_delay)


def wait_for_db(host, port, base_delay=2):
    """Wait until the database is accepting connections."""
    attempt = 0
    while True:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"[DB] Connection to {host}:{port} successful")
                return
        except Exception:
            delay = exponential_backoff(base_delay, attempt)
            print(
                f"[DB] Waiting for {host}:{port}... retry {attempt+1}, next attempt in {delay}s")
            time.sleep(delay)
            attempt += 1


def wait_for_mlflow(url, base_delay=2):
    """Wait until MLflow server is responding with HTTP 200."""
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"[MLflow] MLflow server is up at {url}")
                    return
        except Exception:
            delay = exponential_backoff(base_delay, attempt)
            print(
                f"[MLflow] Waiting for MLflow at {url}... retry {attempt+1}, next attempt in {delay}s")
            time.sleep(delay)
            attempt += 1


if __name__ == "__main__":
    # Use environment variables if available, otherwise defaults
    db_host = os.getenv("DB_HOST", "db")
    db_port = int(os.getenv("DB_PORT", 5432))
    mlflow_url = os.getenv("MLFLOW_URL", "http://mlflow:5000")
    base_delay = float(os.getenv("WAIT_BASE_DELAY", 2))

    print("[INFO] Waiting for dependent services to be ready...")
    wait_for_db(db_host, db_port, base_delay)
    wait_for_mlflow(mlflow_url, base_delay)
    print("[INFO] All services are ready!")
