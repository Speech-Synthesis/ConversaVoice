"""
Fallback Manager for ConversaVoice.

Provides automatic switching between cloud and local services
when cloud APIs fail or are unavailable.
"""

import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Type of service (LLM or TTS)."""
    LLM = "llm"
    TTS = "tts"


class ServiceMode(Enum):
    """Service mode indicating which backend is active."""
    CLOUD = "cloud"
    LOCAL = "local"
    UNAVAILABLE = "unavailable"


@dataclass
class ServiceStatus:
    """Status of a service."""
    mode: ServiceMode = ServiceMode.CLOUD
    cloud_available: bool = True
    local_available: bool = False
    consecutive_failures: int = 0
    last_error: Optional[str] = None


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    # Number of consecutive failures before switching to fallback
    failure_threshold: int = 2
    # Whether to automatically try cloud again after local succeeds
    auto_recover: bool = True
    # Number of successful local calls before trying cloud again
    recovery_threshold: int = 5
    # Whether to prefer local even when cloud is available
    prefer_local: bool = False


class FallbackManager:
    """
    Manages fallback between cloud and local services.

    Tracks service health and automatically switches between
    cloud and local backends based on availability and errors.
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        """
        Initialize fallback manager.

        Args:
            config: Fallback configuration. Uses defaults if not provided.
        """
        self.config = config or FallbackConfig()
        self._services: dict[ServiceType, ServiceStatus] = {
            ServiceType.LLM: ServiceStatus(),
            ServiceType.TTS: ServiceStatus(),
        }
        self._local_success_count: dict[ServiceType, int] = {
            ServiceType.LLM: 0,
            ServiceType.TTS: 0,
        }
        self._on_mode_change: Optional[Callable[[ServiceType, ServiceMode], None]] = None

    def set_mode_change_callback(
        self,
        callback: Callable[[ServiceType, ServiceMode], None]
    ) -> None:
        """
        Set callback for mode changes.

        Args:
            callback: Function called when service mode changes.
        """
        self._on_mode_change = callback

    def get_status(self, service_type: ServiceType) -> ServiceStatus:
        """
        Get current status of a service.

        Args:
            service_type: Type of service (LLM or TTS).

        Returns:
            Current service status.
        """
        return self._services[service_type]

    def get_mode(self, service_type: ServiceType) -> ServiceMode:
        """
        Get current mode of a service.

        Args:
            service_type: Type of service (LLM or TTS).

        Returns:
            Current service mode.
        """
        return self._services[service_type].mode

    def set_cloud_available(self, service_type: ServiceType, available: bool) -> None:
        """
        Set cloud service availability.

        Args:
            service_type: Type of service.
            available: Whether cloud service is available.
        """
        status = self._services[service_type]
        status.cloud_available = available

        if not available and status.mode == ServiceMode.CLOUD:
            self._switch_to_local(service_type)

    def set_local_available(self, service_type: ServiceType, available: bool) -> None:
        """
        Set local service availability.

        Args:
            service_type: Type of service.
            available: Whether local service is available.
        """
        self._services[service_type].local_available = available

    def report_success(self, service_type: ServiceType) -> None:
        """
        Report successful service call.

        Args:
            service_type: Type of service that succeeded.
        """
        status = self._services[service_type]
        status.consecutive_failures = 0
        status.last_error = None

        if status.mode == ServiceMode.LOCAL:
            self._local_success_count[service_type] += 1

            # Try to recover to cloud after threshold
            if (self.config.auto_recover and
                status.cloud_available and
                self._local_success_count[service_type] >= self.config.recovery_threshold):
                self._switch_to_cloud(service_type)

    def report_failure(
        self,
        service_type: ServiceType,
        error: Optional[str] = None
    ) -> ServiceMode:
        """
        Report failed service call.

        Args:
            service_type: Type of service that failed.
            error: Error message or description.

        Returns:
            Current service mode after handling failure.
        """
        status = self._services[service_type]
        status.consecutive_failures += 1
        status.last_error = error

        logger.warning(
            f"{service_type.value} failure #{status.consecutive_failures}: {error}"
        )

        # Switch to local if threshold exceeded and in cloud mode
        if (status.mode == ServiceMode.CLOUD and
            status.consecutive_failures >= self.config.failure_threshold):
            if status.local_available:
                self._switch_to_local(service_type)
            else:
                status.mode = ServiceMode.UNAVAILABLE
                logger.error(f"{service_type.value} unavailable: no local fallback")

        # If local fails, mark as unavailable
        elif status.mode == ServiceMode.LOCAL:
            status.mode = ServiceMode.UNAVAILABLE
            logger.error(f"{service_type.value} unavailable: local fallback also failed")

        return status.mode

    def _switch_to_local(self, service_type: ServiceType) -> None:
        """Switch service to local mode."""
        status = self._services[service_type]
        old_mode = status.mode
        status.mode = ServiceMode.LOCAL
        self._local_success_count[service_type] = 0

        logger.info(f"{service_type.value} switched from {old_mode.value} to local")

        if self._on_mode_change:
            self._on_mode_change(service_type, ServiceMode.LOCAL)

    def _switch_to_cloud(self, service_type: ServiceType) -> None:
        """Switch service back to cloud mode."""
        status = self._services[service_type]
        old_mode = status.mode
        status.mode = ServiceMode.CLOUD
        status.consecutive_failures = 0
        self._local_success_count[service_type] = 0

        logger.info(f"{service_type.value} recovered from {old_mode.value} to cloud")

        if self._on_mode_change:
            self._on_mode_change(service_type, ServiceMode.CLOUD)

    def should_use_local(self, service_type: ServiceType) -> bool:
        """
        Check if local service should be used.

        Args:
            service_type: Type of service.

        Returns:
            True if local service should be used.
        """
        if self.config.prefer_local:
            return self._services[service_type].local_available

        return self._services[service_type].mode == ServiceMode.LOCAL

    def reset(self, service_type: Optional[ServiceType] = None) -> None:
        """
        Reset service status.

        Args:
            service_type: Service to reset. If None, resets all services.
        """
        if service_type:
            self._services[service_type] = ServiceStatus()
            self._local_success_count[service_type] = 0
        else:
            for st in ServiceType:
                self._services[st] = ServiceStatus()
                self._local_success_count[st] = 0

    def get_summary(self) -> dict:
        """
        Get summary of all service statuses.

        Returns:
            Dictionary with service status information.
        """
        return {
            service_type.value: {
                "mode": status.mode.value,
                "cloud_available": status.cloud_available,
                "local_available": status.local_available,
                "consecutive_failures": status.consecutive_failures,
                "last_error": status.last_error,
            }
            for service_type, status in self._services.items()
        }


def with_fallback(
    fallback_manager: FallbackManager,
    service_type: ServiceType,
    cloud_fn: Callable[..., Any],
    local_fn: Callable[..., Any],
    *args,
    **kwargs
) -> Any:
    """
    Execute function with automatic fallback.

    Tries cloud function first, falls back to local on failure.

    Args:
        fallback_manager: FallbackManager instance.
        service_type: Type of service being called.
        cloud_fn: Cloud service function.
        local_fn: Local fallback function.
        *args: Arguments to pass to functions.
        **kwargs: Keyword arguments to pass to functions.

    Returns:
        Result from either cloud or local function.

    Raises:
        Exception: If both cloud and local fail.
    """
    status = fallback_manager.get_status(service_type)

    # Determine which function to try first
    if fallback_manager.should_use_local(service_type):
        primary_fn, secondary_fn = local_fn, cloud_fn
        primary_is_local = True
    else:
        primary_fn, secondary_fn = cloud_fn, local_fn
        primary_is_local = False

    # Try primary
    try:
        result = primary_fn(*args, **kwargs)
        fallback_manager.report_success(service_type)
        return result
    except Exception as e:
        error_msg = str(e)
        fallback_manager.report_failure(service_type, error_msg)

        # If primary was cloud and local is available, try local
        if not primary_is_local and status.local_available:
            try:
                result = secondary_fn(*args, **kwargs)
                fallback_manager.report_success(service_type)
                return result
            except Exception as e2:
                fallback_manager.report_failure(service_type, str(e2))
                raise

        raise
