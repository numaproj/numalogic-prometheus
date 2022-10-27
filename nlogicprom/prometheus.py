import pandas
import logging
import requests
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

APP_QUERIES = {
    "error_count": "namespace_app_pod_http_server_requests_errors{namespace='$namespace'}",
    "error_rate": "namespace_app_pod_http_server_requests_error_rate{namespace='$namespace'}",
    "latency": "namespace_app_pod_http_server_requests_latency{namespace='$namespace'}",
    "cpu": "namespace_asset_pod_cpu_utilization{namespace='$namespace'}",
    "memory": "namespace_asset_pod_memory_utilization{namespace='$namespace'}",
    "hash_error_rate": "namespace_hash_pod_http_server_requests_error_rate{namespace='$namespace'}",
    "hash_latency": "namespace_hash_pod_http_server_requests_latency{namespace='$namespace'}",
    "hash_test_data": "namespace_hash_pod_http_server_requests_error_rate{namespace='$namespace', rollouts_pod_template_hash='7dfdd5f89b'}",
}


class Prometheus:
    def __init__(self, prometheus_server: str):
        self.PROMETHEUS_SERVER = prometheus_server

    def query_metric(
        self,
        metric: str,
        namespace: str,
        start: float,
        end: float,
        hash_col: bool = False,
        step: str = "5s",
    ) -> pd.DataFrame:
        query = APP_QUERIES.get(metric).replace("$namespace", namespace)
        LOGGER.info("Prometheus Query: %s", query)

        results = self.query_range(query, start, end, step)
        data_frames = []
        for result in results:
            arr = np.array(result["values"])
            _df = pd.DataFrame(arr, columns=["timestamp", metric])
            _df = _df.astype(float)
            if hash_col:
                data = result["metric"]
                if "pod_template_hash" in data:
                    hash_val = str(data["pod_template_hash"])
                else:
                    hash_val = str(data["rollouts_pod_template_hash"])
                _df["hash"] = hash_val
            data_frames.append(_df)
        df = pandas.DataFrame()

        if data_frames:
            df = pd.concat(data_frames)

        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index.astype(int), unit="s")
        return df

    def query_range(self, query: str, start: float, end: float, step: str = "5s"):
        results = []
        try:
            response = requests.get(
                self.PROMETHEUS_SERVER + "/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": step},
            )
            print(response)
            results = response.json()["data"]["result"]
        except Exception as ex:
            LOGGER.exception("error: %r", ex)
        return results

    def query(self, query: str):
        results = []
        response = requests.get(self.PROMETHEUS_SERVER + "/api/v1/query", params={"query": query})
        if response:
            results = response.json()["data"]["result"]
        return results
