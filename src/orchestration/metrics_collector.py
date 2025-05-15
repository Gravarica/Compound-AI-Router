import time
from typing import Dict, Any


class MetricsCollector:
    """
    Collects and manages performance metrics.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.router_start_time = None
        self.router_end_time = None
        self.llm_start_time = None
        self.llm_end_time = None
        self.raw_llm_metrics = {}

    def start_query(self):
        """Mark the start of query processing."""
        self.reset()
        self.start_time = time.time()

    def end_query(self):
        """Mark the end of query processing."""
        self.end_time = time.time()

    def start_routing(self):
        """Mark the start of routing."""
        self.router_start_time = time.time()

    def end_routing(self):
        """Mark the end of routing."""
        self.router_end_time = time.time()

    def start_llm(self):
        """Mark the start of LLM processing."""
        self.llm_start_time = time.time()

    def end_llm(self):
        """Mark the end of LLM processing."""
        self.llm_end_time = time.time()

    def set_llm_metrics(self, metrics: Dict[str, Any]):
        """
        Set raw LLM metrics.

        Args:
            metrics: Metrics from the LLM
        """
        self.raw_llm_metrics = metrics

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.

        Returns:
            Dictionary of metrics
        """
        total_time = (self.end_time - self.start_time) * 1000 if self.end_time else None
        router_time = (self.router_end_time - self.router_start_time) * 1000 if self.router_end_time else None
        llm_time = (self.llm_end_time - self.llm_start_time) * 1000 if self.llm_end_time else None

        llm_latency = self.raw_llm_metrics.get('latency_ms', llm_time)

        rate_limit_wait_time = self.raw_llm_metrics.get('rate_limit_wait_time', 0)

        clean_processing_time = total_time - rate_limit_wait_time if total_time else None

        return {
            'total_time_ms': round(total_time, 2) if total_time else None,
            'routing_time_ms': round(router_time, 2) if router_time else None,
            'llm_latency_ms': round(llm_latency, 2) if llm_latency else None,
            'clean_processing_time_ms': round(clean_processing_time, 2) if clean_processing_time else None,
            'rate_limit_wait_time_ms': rate_limit_wait_time,
            'raw_llm_metrics': self.raw_llm_metrics
        }