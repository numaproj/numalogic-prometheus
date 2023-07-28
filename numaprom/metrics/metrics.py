import time

from prometheus_client import start_http_server

def start_metrics_server(port):
    start_http_server(port)
