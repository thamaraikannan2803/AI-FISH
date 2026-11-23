"""
===============================================================================
Fish_AI_Project - Decision Engine Module (decision_engine.py)
===============================================================================
Automated decision engine enforcing strict aquaculture rules:
  IF Water Quality is Safe AND Fish are Hungry -> ACTIVATE VIRTUAL FEEDER
  ELSE -> STOP/DAMPEN FEEDING & LOG REASON TO DATABASE.
Calculates exact feed amount in grams using fish count and growth stage formulas.
"""

import time
import logging
from typing import Dict, Any
from config import FEED_RATES_GRAMS, MAX_FEED_PER_SESSION_GRAMS

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Evaluates multi-sensor telemetry and triggers automated feeding decisions."""

    def __init__(self) -> None:
        """Initialize decision engine."""
        self.last_decision_time = 0.0
        self.cooldown_sec = 30.0  # Prevent back-to-back overfeeding

    def evaluate(
        self,
        water_telemetry: Dict[str, Any],
        hunger_prediction: Dict[str, Any],
        fish_count: int,
        growth_stage: str = "Fingerling"
    ) -> Dict[str, Any]:
        """Evaluate system state and generate feeding decision.

        Args:
            water_telemetry: Dict from water_quality.py.
            hunger_prediction: Dict from hunger_prediction.py.
            fish_count: Active count of detected fish.
            growth_stage: Dominant fish growth stage.

        Returns:
            Dict containing should_feed boolean, recommended_grams, reason, and status.
        """
        now = time.time()
        is_water_safe = water_telemetry.get("is_safe", False)
        hunger_status = hunger_prediction.get("status", "Unknown")
        hunger_score = hunger_prediction.get("score", 0.0)

        # Check safety & hunger conditions
        is_hungry = hunger_status in ["Hungry", "Feeding"] and hunger_score >= 55.0

        if not is_water_safe:
            alerts = water_telemetry.get("alerts", ["Water quality out of safe range"])
            reason = f"FEEDING BLOCKED: Water Quality Unsafe ({'; '.join(alerts)})"
            return {
                "should_feed": False,
                "recommended_grams": 0.0,
                "reason": reason,
                "water_quality_status": "Unsafe",
                "hunger_status": hunger_status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        if not is_hungry:
            reason = f"FEEDING IDLE: Fish are {hunger_status} (Hunger Score: {hunger_score:.1f}/100)"
            return {
                "should_feed": False,
                "recommended_grams": 0.0,
                "reason": reason,
                "water_quality_status": "Optimal",
                "hunger_status": hunger_status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        # Cooldown check to prevent overfeeding
        if (now - self.last_decision_time) < self.cooldown_sec:
            time_left = int(self.cooldown_sec - (now - self.last_decision_time))
            reason = f"FEEDING PAUSED: System in post-feed digestion cooldown ({time_left}s remaining)"
            return {
                "should_feed": False,
                "recommended_grams": 0.0,
                "reason": reason,
                "water_quality_status": "Optimal",
                "hunger_status": hunger_status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        # Calculate Feed Dosage (grams)
        # Dosage = (Rate per fish for stage) * fish_count
        rate_per_fish = FEED_RATES_GRAMS.get(growth_stage, 0.35)
        raw_dosage = rate_per_fish * max(1, fish_count)
        dosage = min(MAX_FEED_PER_SESSION_GRAMS, round(raw_dosage, 2))

        self.last_decision_time = now
        reason = f"FEEDING APPROVED: Water Quality Safe & Fish Hungry (Count: {fish_count} {growth_stage}s | Dosage: {dosage}g)"

        logger.info(reason)

        return {
            "should_feed": True,
            "recommended_grams": dosage,
            "reason": reason,
            "water_quality_status": "Optimal",
            "hunger_status": hunger_status,
            "growth_stage": growth_stage,
            "fish_count": fish_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def reset_cooldown(self) -> None:
        """Reset the post-feed cooldown timer (useful for testing or manual override)."""
        self.last_decision_time = 0.0
        logger.info("DecisionEngine cooldown timer reset.")

    def get_decision_summary(
        self,
        water_telemetry: Dict[str, Any],
        hunger_prediction: Dict[str, Any],
        fish_count: int,
        growth_stage: str = "Fingerling"
    ) -> str:
        """Return a compact single-line summary of the feeding decision.

        Args:
            water_telemetry: Dict from water_quality.py.
            hunger_prediction: Dict from hunger_prediction.py.
            fish_count: Active fish count.
            growth_stage: Growth stage string.

        Returns:
            One-line human-readable decision string.
        """
        result = self.evaluate(water_telemetry, hunger_prediction, fish_count, growth_stage)
        icon = "🟢 FEED" if result["should_feed"] else "🔴 HOLD"
        grams = f" | {result['recommended_grams']}g" if result["should_feed"] else ""
        return f"[{icon}{grams}] {result['reason']}"


# ===========================================================================
# Standalone Demo — Module 8
# ===========================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 65)
    print("  MODULE 8 — Decision Engine (Standalone Demo)")
    print("=" * 65)

    engine = DecisionEngine()

    # --- Shared water/hunger fixtures ---
    SAFE_WATER = {
        "ph": 7.4, "temp_c": 26.5, "turbidity_ntu": 4.0,
        "dissolved_oxygen_mg_l": 7.5, "stress_index": 0.05,
        "is_safe": True, "alerts": []
    }
    UNSAFE_WATER = {
        "ph": 5.5, "temp_c": 35.0, "turbidity_ntu": 25.0,
        "dissolved_oxygen_mg_l": 2.5, "stress_index": 0.80,
        "is_safe": False,
        "alerts": ["Critical pH 5.5", "Hypoxia DO 2.5 mg/L"]
    }
    HUNGRY = {"status": "Hungry",   "score": 88.0, "confidence": 0.95}
    SATIATED = {"status": "Satiated", "score": 20.0, "confidence": 0.87}

    scenarios = [
        ("Safe water + Hungry fish",    SAFE_WATER,   HUNGRY,   8,  "Fingerling"),
        ("Safe water + Satiated fish",  SAFE_WATER,   SATIATED, 8,  "Fingerling"),
        ("Unsafe water + Hungry fish",  UNSAFE_WATER, HUNGRY,   8,  "Fingerling"),
        ("Adult fish (high dosage)",    SAFE_WATER,   HUNGRY,   12, "Adult"),
        ("Fry fish (micro dosage)",     SAFE_WATER,   HUNGRY,   20, "Fry"),
    ]

    print(f"\n  {'Scenario':<35} {'Feed?':<8} {'Grams':<10} {'Status'}")
    print("  " + "─" * 65)
    for name, water, hunger, count, stage in scenarios:
        engine.reset_cooldown()
        result = engine.evaluate(water, hunger, count, stage)
        feed_icon = "✓ YES" if result["should_feed"] else "✗ NO"
        grams = f"{result['recommended_grams']}g" if result["should_feed"] else "—"
        print(f"  {name:<35} {feed_icon:<8} {grams:<10} {result.get('water_quality_status','—')}")

    # --- Cooldown test ---
    print(f"\n  [Cooldown Test]")
    engine.reset_cooldown()
    r1 = engine.evaluate(SAFE_WATER, HUNGRY, 8)
    r2 = engine.evaluate(SAFE_WATER, HUNGRY, 8)   # Should hit cooldown
    print(f"  First call  → Feed: {r1['should_feed']} | {r1['reason'][:55]}…")
    print(f"  Second call → Feed: {r2['should_feed']} | {r2['reason'][:55]}…")

    # --- Dosage across growth stages ---
    print(f"\n  [Dosage by Growth Stage — 10 fish]")
    for stage in ["Fry", "Fingerling", "Juvenile", "Adult"]:
        engine.reset_cooldown()
        r = engine.evaluate(SAFE_WATER, HUNGRY, fish_count=10, growth_stage=stage)
        print(f"  {stage:<12} → {r['recommended_grams']}g")

    print(f"\n{'='*65}")
    print("  MODULE 8 DEMO COMPLETE — Decision engine fully operational.")
    print(f"{'='*65}")
