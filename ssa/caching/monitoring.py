"""
Cache monitoring and alerting functionality.

This module provides monitoring capabilities for the prompt cache system,
including health checks, performance alerts, and usage analytics.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .prompt_cache import get_global_cache

logger = logging.getLogger(__name__)


@dataclass
class CacheAlert:
    """Represents a cache-related alert."""

    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: float
    metrics: Dict[str, Any]
    threshold: Optional[float] = None
    current_value: Optional[float] = None


class CacheMonitor:
    """
    Monitor cache performance and generate alerts.

    This class tracks cache metrics over time and can trigger alerts
    when certain thresholds are exceeded.
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[CacheAlert], None]] = None,
        log_alerts: bool = True,
    ):
        """
        Initialize cache monitor.

        Args:
            alert_callback: Optional callback function for alerts
            log_alerts: Whether to log alerts to the logger
        """
        self.alert_callback = alert_callback
        self.log_alerts = log_alerts
        self.last_check_time = time.time()
        self.alert_history: List[CacheAlert] = []

        # Default thresholds
        self.thresholds = {
            "hit_rate_low": 0.1,  # Alert if hit rate below 10%
            "utilization_high": 0.9,  # Alert if utilization above 90%
            "expired_ratio_high": 0.3,  # Alert if >30% entries are expired
            "efficiency_score_low": 0.05,  # Alert if efficiency score below 5%
        }

    def set_threshold(self, metric: str, value: float) -> None:
        """Set a custom threshold for alerts."""
        self.thresholds[metric] = value
        logger.debug(f"Set threshold {metric} = {value}")

    def check_cache_health(self) -> List[CacheAlert]:
        """
        Check cache health and return any alerts.

        Returns:
            List of alerts generated
        """
        alerts = []
        cache = get_global_cache()

        if not cache:
            alert = CacheAlert(
                alert_type="cache_unavailable",
                severity="critical",
                message="Global cache is not initialized",
                timestamp=time.time(),
                metrics={},
            )
            alerts.append(alert)
            self._handle_alert(alert)
            return alerts

        try:
            stats = cache.get_stats()
            current_time = time.time()

            # Check hit rate
            if stats.get("total_requests", 0) > 10:  # Only check if we have enough data
                hit_rate = stats.get("hit_rate", 0.0)
                if hit_rate < self.thresholds["hit_rate_low"]:
                    alert = CacheAlert(
                        alert_type="low_hit_rate",
                        severity="medium",
                        message=f"Cache hit rate is low: {hit_rate:.2%}",
                        timestamp=current_time,
                        metrics=stats,
                        threshold=self.thresholds["hit_rate_low"],
                        current_value=hit_rate,
                    )
                    alerts.append(alert)

            # Check utilization
            utilization = stats.get("utilization", 0.0)
            if utilization > self.thresholds["utilization_high"]:
                alert = CacheAlert(
                    alert_type="high_utilization",
                    severity="high",
                    message=f"Cache utilization is high: {utilization:.2%}",
                    timestamp=current_time,
                    metrics=stats,
                    threshold=self.thresholds["utilization_high"],
                    current_value=utilization,
                )
                alerts.append(alert)

            # Check expired entries ratio
            expired_ratio = stats.get("expired_entries_ratio", 0.0)
            if expired_ratio > self.thresholds["expired_ratio_high"]:
                alert = CacheAlert(
                    alert_type="high_expired_ratio",
                    severity="medium",
                    message=f"High ratio of expired entries: {expired_ratio:.2%}",
                    timestamp=current_time,
                    metrics=stats,
                    threshold=self.thresholds["expired_ratio_high"],
                    current_value=expired_ratio,
                )
                alerts.append(alert)

            # Check efficiency score
            efficiency = stats.get("cache_efficiency_score", 0.0)
            if (
                efficiency < self.thresholds["efficiency_score_low"]
                and stats.get("total_requests", 0) > 5
            ):
                alert = CacheAlert(
                    alert_type="low_efficiency",
                    severity="medium",
                    message=f"Cache efficiency is low: {efficiency:.3f}",
                    timestamp=current_time,
                    metrics=stats,
                    threshold=self.thresholds["efficiency_score_low"],
                    current_value=efficiency,
                )
                alerts.append(alert)

            # Check if cache is healthy overall
            if not stats.get("is_healthy", True):
                alert = CacheAlert(
                    alert_type="unhealthy_cache",
                    severity="high",
                    message="Cache is marked as unhealthy based on multiple metrics",
                    timestamp=current_time,
                    metrics=stats,
                )
                alerts.append(alert)

            # Handle all generated alerts
            for alert in alerts:
                self._handle_alert(alert)

            self.last_check_time = current_time

        except Exception as e:
            alert = CacheAlert(
                alert_type="monitoring_error",
                severity="high",
                message=f"Failed to check cache health: {e}",
                timestamp=time.time(),
                metrics={},
            )
            alerts.append(alert)
            self._handle_alert(alert)

        return alerts

    def _handle_alert(self, alert: CacheAlert) -> None:
        """Handle a generated alert."""
        # Add to history
        self.alert_history.append(alert)

        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

        # Log alert
        if self.log_alerts:
            log_level = {
                "low": logging.INFO,
                "medium": logging.WARNING,
                "high": logging.ERROR,
                "critical": logging.CRITICAL,
            }.get(alert.severity, logging.WARNING)

            logger.log(log_level, f"Cache Alert [{alert.alert_type}]: {alert.message}")

        # Call custom callback
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_alert_history(self, hours: Optional[float] = None) -> List[CacheAlert]:
        """
        Get alert history, optionally filtered by time.

        Args:
            hours: If provided, only return alerts from the last N hours

        Returns:
            List of alerts
        """
        if hours is None:
            return self.alert_history.copy()

        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive cache monitoring report."""
        cache = get_global_cache()
        current_time = time.time()

        if not cache:
            return {
                "status": "error",
                "message": "No global cache available",
                "timestamp": current_time,
            }

        try:
            stats = cache.get_stats()
            recent_alerts = self.get_alert_history(hours=24)

            # Categorize alerts by severity
            alert_counts = {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0,
            }

            for alert in recent_alerts:
                alert_counts[alert.severity] += 1

            # Calculate performance trends (simplified)
            performance_score = 0.0
            if stats.get("total_requests", 0) > 0:
                performance_score = (
                    stats.get("hit_rate", 0.0) * 0.4
                    + (1 - stats.get("utilization", 0.0)) * 0.3
                    + stats.get("cache_efficiency_score", 0.0) * 0.3
                )

            return {
                "status": "ok",
                "timestamp": current_time,
                "cache_stats": stats,
                "alerts_24h": alert_counts,
                "total_alerts_24h": len(recent_alerts),
                "performance_score": performance_score,
                "health_status": (
                    "healthy" if stats.get("is_healthy", False) else "unhealthy"
                ),
                "recommendations": self._generate_recommendations(stats, recent_alerts),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to generate report: {e}",
                "timestamp": current_time,
            }

    def _generate_recommendations(
        self, stats: Dict[str, Any], recent_alerts: List[CacheAlert]
    ) -> List[str]:
        """Generate recommendations based on cache performance."""
        recommendations = []

        # Hit rate recommendations
        hit_rate = stats.get("hit_rate", 0.0)
        if hit_rate < 0.2:
            recommendations.append(
                "Consider increasing cache TTL or capacity to improve hit rate"
            )

        # Utilization recommendations
        utilization = stats.get("utilization", 0.0)
        if utilization > 0.8:
            recommendations.append(
                "Consider increasing cache capacity to reduce evictions"
            )
        elif utilization < 0.3 and stats.get("total_requests", 0) > 100:
            recommendations.append(
                "Cache capacity might be too large for current usage patterns"
            )

        # TTL recommendations
        expired_ratio = stats.get("expired_entries_ratio", 0.0)
        if expired_ratio > 0.4:
            recommendations.append(
                "Consider reducing TTL to prevent accumulation of expired entries"
            )

        # Alert-based recommendations
        alert_types = {alert.alert_type for alert in recent_alerts}
        if "high_utilization" in alert_types:
            recommendations.append(
                "Frequent high utilization alerts suggest need for capacity increase"
            )

        if not recommendations:
            recommendations.append(
                "Cache performance looks good - no specific recommendations"
            )

        return recommendations


# Global monitor instance with thread safety
_global_monitor: Optional[CacheMonitor] = None
_monitor_lock = threading.RLock()


def get_global_monitor() -> Optional[CacheMonitor]:
    """Get the global cache monitor instance."""
    with _monitor_lock:
        return _global_monitor


def init_cache_monitoring(
    alert_callback: Optional[Callable[[CacheAlert], None]] = None,
    log_alerts: bool = True,
) -> CacheMonitor:
    """
    Initialize global cache monitoring.

    Args:
        alert_callback: Optional callback for alerts
        log_alerts: Whether to log alerts

    Returns:
        CacheMonitor instance
    """
    global _global_monitor
    with _monitor_lock:
        _global_monitor = CacheMonitor(
            alert_callback=alert_callback,
            log_alerts=log_alerts,
        )
        logger.info("Cache monitoring initialized")
        return _global_monitor


def check_cache_health() -> List[CacheAlert]:
    """Check cache health using global monitor."""
    monitor = get_global_monitor()
    if monitor:
        return monitor.check_cache_health()
    return []


def get_cache_report() -> Dict[str, Any]:
    """Get cache monitoring report."""
    monitor = get_global_monitor()
    if monitor:
        return monitor.generate_report()
    return {
        "status": "error",
        "message": "Cache monitoring not initialized",
        "timestamp": time.time(),
    }
