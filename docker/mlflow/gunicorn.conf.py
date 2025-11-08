import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

# Process naming
proc_name = "mlflow"
pythonpath = "/app"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment if using SSL)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Server hooks
def pre_fork(server, worker):
    server.log.info("Worker %s forking", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker %s forked", worker.pid)