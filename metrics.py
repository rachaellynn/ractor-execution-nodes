"""
Payload size tracking and metrics collection for Ractor execution agents.
Simplified version for deployment in Ractor agents.
"""
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class PayloadMeasurement:
    """Single measurement of data in transit"""
    timestamp: str
    operation: str
    payload_size_bytes: int
    payload_type: str
    task_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PayloadTracker:
    """Lightweight payload tracker for Ractor agents"""

    def __init__(self):
        self._measurements: List[PayloadMeasurement] = []

    def measure_payload(
        self,
        data: Any,
        operation: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Measure payload size and record it"""
        payload_size = self._calculate_size(data)
        payload_type = self._detect_type(data)

        measurement = PayloadMeasurement(
            timestamp=datetime.utcnow().isoformat(),
            operation=operation,
            payload_size_bytes=payload_size,
            payload_type=payload_type,
            task_id=task_id,
            metadata=metadata or {}
        )

        self._measurements.append(measurement)
        return payload_size

    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes"""
        if data is None:
            return 0

        if isinstance(data, str):
            return len(data.encode('utf-8'))
        elif isinstance(data, bytes):
            return len(data)
        elif isinstance(data, (dict, list)):
            return len(json.dumps(data, default=str).encode('utf-8'))
        else:
            return len(str(data).encode('utf-8'))

    def _detect_type(self, data: Any) -> str:
        """Detect the type of payload"""
        if isinstance(data, str):
            return "text"
        elif isinstance(data, bytes):
            return "binary"
        elif isinstance(data, (dict, list)):
            return "json"
        else:
            return "other"

    def get_measurements(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get measurements, optionally filtered by task_id"""
        measurements = self._measurements.copy()

        if task_id:
            measurements = [m for m in measurements if m.task_id == task_id]

        return [asdict(m) for m in measurements]

    def get_summary_stats(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics"""
        measurements = self.get_measurements(task_id)

        if not measurements:
            return {"total_bytes": 0, "count": 0, "operation_count": 0}

        sizes = [m["payload_size_bytes"] for m in measurements]

        return {
            "total_bytes": sum(sizes),
            "count": len(sizes),
            "operation_count": len(measurements),
            "avg_bytes": sum(sizes) / len(sizes) if sizes else 0,
            "max_bytes": max(sizes) if sizes else 0,
            "min_bytes": min(sizes) if sizes else 0,
            "operations": list(set(m["operation"] for m in measurements))
        }

    def clear(self):
        """Clear all measurements"""
        self._measurements.clear()


# Global tracker instance
payload_tracker = PayloadTracker()


def track_payload(
    data: Any,
    operation: str,
    task_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """Convenience function to track payload"""
    return payload_tracker.measure_payload(
        data=data,
        operation=operation,
        task_id=task_id,
        metadata=metadata
    )


class PayloadTrackingMixin:
    """Mixin for adding payload tracking to execution classes"""

    def track_input(self, data: Any, operation: str, task_id: Optional[str] = None) -> int:
        """Track input payload"""
        return track_payload(
            data=data,
            operation=f"{operation}_input",
            task_id=task_id,
            metadata={"component": self.__class__.__name__.lower()}
        )

    def track_output(self, data: Any, operation: str, task_id: Optional[str] = None) -> int:
        """Track output payload"""
        return track_payload(
            data=data,
            operation=f"{operation}_output",
            task_id=task_id,
            metadata={"component": self.__class__.__name__.lower()}
        )