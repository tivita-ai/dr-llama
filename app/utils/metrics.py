import time
from functools import wraps
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram

from app.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    def __init__(self):
        # Request metrics
        self.request_counter = Counter(
            "model_requests_total",
            "Total number of requests to the model",
            ["endpoint", "status"],
        )
        self.request_latency = Histogram(
            "model_request_latency_seconds",
            "Time spent processing request",
            ["endpoint"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")),
        )

        # Model metrics
        self.token_counter = Counter(
            "model_tokens_total",
            "Total number of tokens processed",
            ["operation"],
        )
        self.model_memory = Gauge(
            "model_memory_bytes",
            "Current memory usage of the model",
            ["device"],
        )

        # System metrics
        self.gpu_memory_usage = Gauge(
            "gpu_memory_usage_bytes",
            "GPU memory usage",
            ["device"],
        )
        self.gpu_utilization = Gauge(
            "gpu_utilization_percent",
            "GPU utilization percentage",
            ["device"],
        )

    def track_request(self, endpoint: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    self.request_counter.labels(endpoint=endpoint, status="success").inc()
                    return result
                except Exception as e:
                    self.request_counter.labels(endpoint=endpoint, status="error").inc()
                    logger.error("Request failed: %s", e)
                    raise
                finally:
                    self.request_latency.labels(endpoint=endpoint).observe(time.time() - start_time)

            return wrapper

        return decorator

    def track_tokens(self, operation: str, num_tokens: int):
        self.token_counter.labels(operation=operation).inc(num_tokens)

    def update_model_memory(self, device: str, bytes_used: int):
        self.model_memory.labels(device=device).set(bytes_used)

    def update_gpu_metrics(self, device: str, memory_used: int, utilization: float):
        self.gpu_memory_usage.labels(device=device).set(memory_used)
        self.gpu_utilization.labels(device=device).set(utilization)


metrics = MetricsCollector()
