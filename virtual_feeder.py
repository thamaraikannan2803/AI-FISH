"""
===============================================================================
Fish_AI_Project - Virtual Feeder Module (virtual_feeder.py)
===============================================================================
Software-only virtual feeder simulator requiring no physical hardware.
Manages pellet dispensing animation timers, feed amount counters, feed completion
states, and pellet hopper capacity notifications.
"""

import time
import logging
from typing import Dict, Any
from config import FEEDING_DISPENSE_DURATION_SEC, MAX_FEED_PER_SESSION_GRAMS

logger = logging.getLogger(__name__)


class VirtualFeeder:
    """Software-only Virtual Automated Fish Feeder."""

    def __init__(self, hopper_capacity_grams: float = 2000.0) -> None:
        """Initialize virtual feeder state.

        Args:
            hopper_capacity_grams: Total food pellet capacity in hopper.
        """
        self.capacity_grams = hopper_capacity_grams
        self.current_hopper_level = hopper_capacity_grams
        self.status = "Idle"  # Idle, Dispensing, Completed, Refill Required
        self.target_feed_grams = 0.0
        self.dispensed_session_grams = 0.0
        self.total_dispensed_today = 0.0

        self.dispense_start_time = 0.0
        self.dispense_duration_sec = FEEDING_DISPENSE_DURATION_SEC

    def trigger_dispense(self, amount_grams: float) -> Dict[str, Any]:
        """Trigger a software feeding dispense session.

        Args:
            amount_grams: Requested food dosage in grams.

        Returns:
            Dict representing initial feeder status.
        """
        if self.current_hopper_level < amount_grams:
            self.status = "Refill Required"
            logger.warning("Virtual Feeder Hopper Empty! Refill Required.")
            return self.get_status()

        self.target_feed_grams = min(MAX_FEED_PER_SESSION_GRAMS, amount_grams)
        self.dispensed_session_grams = 0.0
        self.status = "Dispensing"
        self.dispense_start_time = time.time()

        logger.info(f"Virtual Feeder Dispensing Started: {self.target_feed_grams}g")
        return self.get_status()

    def update(self) -> Dict[str, Any]:
        """Update virtual feeder progress timer and pellet count."""
        if self.status != "Dispensing":
            return self.get_status()

        elapsed = time.time() - self.dispense_start_time

        if elapsed >= self.dispense_duration_sec:
            # Dispense complete
            self.dispensed_session_grams = self.target_feed_grams
            self.current_hopper_level = max(0.0, self.current_hopper_level - self.target_feed_grams)
            self.total_dispensed_today += self.target_feed_grams
            self.status = "Completed"
            logger.info(f"Virtual Feeder Dispense Completed: {self.target_feed_grams}g dispensed.")
        else:
            # Partial progress ratio
            progress = elapsed / self.dispense_duration_sec
            self.dispensed_session_grams = round(self.target_feed_grams * progress, 2)

        return self.get_status()

    def refill_hopper(self, amount_grams: float = 2000.0) -> None:
        """Refill food pellet hopper."""
        self.current_hopper_level = min(self.capacity_grams, self.current_hopper_level + amount_grams)
        if self.status == "Refill Required":
            self.status = "Idle"
        logger.info(f"Virtual Feeder Hopper refilled to {self.current_hopper_level}g.")

    def get_status(self) -> Dict[str, Any]:
        """Get current feeder status metrics."""
        elapsed = time.time() - self.dispense_start_time if self.status == "Dispensing" else 0.0
        time_remaining = max(0.0, self.dispense_duration_sec - elapsed) if self.status == "Dispensing" else 0.0

        return {
            "status": self.status,
            "target_feed_grams": self.target_feed_grams,
            "dispensed_session_grams": round(self.dispensed_session_grams, 2),
            "total_dispensed_today": round(self.total_dispensed_today, 2),
            "hopper_level_grams": round(self.current_hopper_level, 1),
            "time_remaining_sec": round(time_remaining, 1),
            "progress_percent": round((self.dispensed_session_grams / max(1.0, self.target_feed_grams)) * 100.0, 1)
        }
