"""
JSON logger for Expera AI.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np


class JSONLogger:
    """
    JSON file logging.

    Provides:
    - Structured JSON output
    - Flexible schema
    - Append mode for arrays
    """

    def __init__(
        self,
        path: str = "logs/metrics.json",
        indent: int = 2,
    ):
        self.path = Path(path)
        self.indent = indent

        # Create directory
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self.data: Dict[str, Any] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        """Load existing data."""
        try:
            with open(self.path) as f:
                self.data = json.load(f)
        except json.JSONDecodeError:
            self.data = {}

    def log(
        self,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Log metrics to JSON.

        Args:
            metrics: Metrics dictionary
        """
        # Add timestamp
        timestamp = datetime.now().isoformat()

        # Get step
        step = metrics.get("step", 0)

        # Determine type
        log_type = metrics.get("type", "metrics")

        if log_type == "metrics":
            # Store in steps dict
            if "steps" not in self.data:
                self.data["steps"] = {}

            self.data["steps"][str(step)] = {
                "timestamp": timestamp,
                **metrics,
            }
        elif log_type == "event":
            # Store in events list
            if "events" not in self.data:
                self.data["events"] = []

            self.data["events"].append({
                "timestamp": timestamp,
                **metrics,
            })

        # Save
        self._save()

    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Log step metrics.

        Args:
            step: Training step
            metrics: Metrics dict
        """
        self.log({**metrics, "step": step, "type": "metrics"})

    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Log event.

        Args:
            event_type: Type of event
            data: Event data
        """
        self.log({**data, "type": event_type, "event_type": event_type})

    def log_checkpoint(
        self,
        step: int,
        path: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log checkpoint save.

        Args:
            step: Training step
            path: Checkpoint path
            metrics: Metrics
        """
        self.log({
            "type": "checkpoint",
            "step": step,
            "path": path,
            "metrics": metrics,
        })

    def _save(self) -> None:
        """Save data to JSON."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=self.indent)

    def read(self) -> Dict[str, Any]:
        """
        Read all logged data.

        Returns:
            Logged data
        """
        return self.data

    def get_steps(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get logged steps.

        Args:
            last_n: Number of recent steps (None for all)

        Returns:
            List of step metrics
        """
        steps = self.data.get("steps", {})

        # Convert to list and sort
        step_list = [
            {"step": int(k), **v}
            for k, v in steps.items()
        ]
        step_list.sort(key=lambda x: x["step"])

        if last_n:
            return step_list[-last_n:]

        return step_list

    def get_recent(self, n: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent metrics.

        Args:
            n: Number of recent entries

        Returns:
            List of recent metrics
        """
        return self.get_steps(last_n=n)

    def get_best(self) -> Optional[Dict[str, Any]]:
        """
        Get best step (lowest loss).

        Returns:
            Best step metrics
        """
        steps = self.get_steps()
        if not steps:
            return None

        # Find min loss
        best = min(steps, key=lambda x: x.get("loss", float("inf")))
        return best

    def close(self) -> None:
        """Close logger."""
        self._save()


class NDJSONLogger:
    """
    Newline-delimited JSON logger for streaming writes.
    """

    def __init__(
        self,
        path: str = "logs/metrics.ndjson",
    ):
        self.path = Path(path)

        # Create directory
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Log metrics (one line per log).

        Args:
            metrics: Metrics dictionary
        """
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()

        # Write as single line
        with open(self.path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def read(self) -> List[Dict[str, Any]]:
        """
        Read all logs.

        Returns:
            List of logged metrics
        """
        if not self.path.exists():
            return []

        logs = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))

        return logs

    def close(self) -> None:
        """Close logger."""
        pass