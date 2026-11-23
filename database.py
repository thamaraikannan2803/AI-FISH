"""
===============================================================================
Fish_AI_Project - Database Manager Module (database.py)
===============================================================================
Manages SQLite database storage for fish detection events, water quality logs,
hunger predictions, feeding decisions, and system event diagnostics.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from config import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite Database Interface for Aquaculture Telemetry."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        """Initialize database connection and construct schema tables.

        Args:
            db_path: Absolute or relative path to SQLite file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_tables()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_tables(self) -> None:
        """Construct database schema if not already present."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Fish Detection Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fish_detection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    fish_count INTEGER NOT NULL,
                    growth_stage TEXT NOT NULL,
                    avg_confidence REAL NOT NULL,
                    avg_speed REAL NOT NULL,
                    feeding_zone_count INTEGER NOT NULL
                )
            """)

            # Water Quality Telemetry Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS water_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ph REAL NOT NULL,
                    temperature REAL NOT NULL,
                    turbidity REAL NOT NULL,
                    dissolved_oxygen REAL NOT NULL,
                    is_safe INTEGER NOT NULL,
                    stress_index REAL NOT NULL,
                    alerts TEXT
                )
            """)

            # Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    hunger_status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    hunger_score REAL NOT NULL,
                    predictor_engine TEXT NOT NULL,
                    suppressed_by_env INTEGER NOT NULL,
                    primary_factors TEXT
                )
            """)

            # Feeding History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feeding_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    should_feed INTEGER NOT NULL,
                    amount_grams REAL NOT NULL,
                    growth_stage TEXT NOT NULL,
                    fish_count INTEGER NOT NULL,
                    hunger_status TEXT NOT NULL,
                    water_quality_status TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
            """)

            # System Logs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    module TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            """)

            conn.commit()
            logger.info("SQLite Database Schema initialized successfully.")

    def log_detection(self, fish_count: int, growth_stage: str, avg_conf: float, avg_speed: float, zone_count: int) -> None:
        """Insert fish detection event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fish_detection (timestamp, fish_count, growth_stage, avg_confidence, avg_speed, feeding_zone_count)
                VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?)
            """, (fish_count, growth_stage, avg_conf, avg_speed, zone_count))
            conn.commit()

    def log_water_quality(self, ph: float, temp: float, turbidity: float, do: float, is_safe: bool, stress_index: float, alerts: str) -> None:
        """Insert water quality telemetry frame."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO water_quality (timestamp, ph, temperature, turbidity, dissolved_oxygen, is_safe, stress_index, alerts)
                VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?)
            """, (ph, temp, turbidity, do, 1 if is_safe else 0, stress_index, alerts))
            conn.commit()

    def log_prediction(self, status: str, conf: float, score: float, engine: str, suppressed: bool, factors: str) -> None:
        """Insert hunger prediction event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (timestamp, hunger_status, confidence, hunger_score, predictor_engine, suppressed_by_env, primary_factors)
                VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?)
            """, (status, conf, score, engine, 1 if suppressed else 0, factors))
            conn.commit()

    def log_feeding_decision(self, should_feed: bool, amount: float, stage: str, count: int, hunger_status: str, wq_status: str, reason: str) -> None:
        """Insert automated feeding decision."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feeding_history (timestamp, should_feed, amount_grams, growth_stage, fish_count, hunger_status, water_quality_status, reason)
                VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?)
            """, (1 if should_feed else 0, amount, stage, count, hunger_status, wq_status, reason))
            conn.commit()

    def log_system_event(self, level: str, module: str, message: str) -> None:
        """Insert diagnostic log entry."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_logs (timestamp, level, module, message)
                VALUES (datetime('now', 'localtime'), ?, ?, ?)
            """, (level, module, message))
            conn.commit()

    def get_recent_feedings(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent feeding history records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feeding_history ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_water_quality(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent water quality records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM water_quality ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_system_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch system diagnostic logs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_logs ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent hunger prediction records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_detections(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent fish detection event records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM fish_detection ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Fetch high-level telemetry KPIs and summary metrics across all tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(amount_grams), 0.0) FROM feeding_history WHERE should_feed = 1")
            feed_count, total_grams = cursor.fetchone()

            cursor.execute("SELECT COALESCE(AVG(fish_count), 0.0), COALESCE(AVG(avg_speed), 0.0) FROM fish_detection")
            avg_fish, avg_speed = cursor.fetchone()

            cursor.execute("SELECT COALESCE(AVG(dissolved_oxygen), 0.0), COALESCE(AVG(temperature), 0.0), COALESCE(AVG(ph), 0.0) FROM water_quality")
            avg_do, avg_temp, avg_ph = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE hunger_status = 'Hungry'")
            hungry_count = cursor.fetchone()[0]

            return {
                "total_feedings": int(feed_count or 0),
                "total_dispensed_grams": round(float(total_grams or 0.0), 2),
                "avg_fish_count": round(float(avg_fish or 0.0), 1),
                "avg_speed": round(float(avg_speed or 0.0), 2),
                "avg_dissolved_oxygen": round(float(avg_do or 0.0), 2),
                "avg_temperature": round(float(avg_temp or 0.0), 1),
                "avg_ph": round(float(avg_ph or 0.0), 2),
                "hungry_event_count": int(hungry_count or 0)
            }

    def seed_demo_data(self, num_records: int = 40) -> None:
        """Seed realistic historical telemetry across tables for academic demonstrations."""
        import random
        from datetime import datetime, timedelta

        stages = ["Fry", "Fingerling", "Juvenile", "Adult"]
        now = datetime.now()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            for i in range(num_records):
                past_time = (now - timedelta(minutes=(num_records - i) * 12)).strftime("%Y-%m-%d %H:%M:%S")
                stage = random.choice(stages)
                count = random.randint(4, 14)
                speed = random.uniform(30.0, 110.0)
                conf = random.uniform(0.75, 0.96)
                zone_count = random.randint(1, count)

                # Detection
                cursor.execute("""
                    INSERT INTO fish_detection (timestamp, fish_count, growth_stage, avg_confidence, avg_speed, feeding_zone_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (past_time, count, stage, round(conf, 2), round(speed, 2), zone_count))

                # Water Quality
                ph = round(random.uniform(7.1, 7.8), 2)
                temp = round(random.uniform(25.0, 27.5), 1)
                turb = round(random.uniform(2.5, 6.0), 1)
                do = round(random.uniform(6.2, 8.4), 2)
                cursor.execute("""
                    INSERT INTO water_quality (timestamp, ph, temperature, turbidity, dissolved_oxygen, is_safe, stress_index, alerts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (past_time, ph, temp, turb, do, 1, 0.05, ""))

                # Predictions & Feeding
                is_hungry = speed > 65.0
                status = "Hungry" if is_hungry else ("Feeding" if speed > 45.0 else "Satiated")
                score = round(random.uniform(65.0, 92.0) if is_hungry else random.uniform(20.0, 48.0), 1)
                cursor.execute("""
                    INSERT INTO predictions (timestamp, hunger_status, confidence, hunger_score, predictor_engine, suppressed_by_env, primary_factors)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (past_time, status, round(random.uniform(0.85, 0.98), 2), score, "RandomForest Classifier (ML)", 0, f"Speed: {speed:.1f} px/s; Zone: {zone_count}/{count}"))

                if is_hungry and i % 3 == 0:
                    dosage = round(count * (0.05 if stage == "Fry" else (0.35 if stage == "Fingerling" else (1.2 if stage == "Juvenile" else 4.5))), 2)
                    cursor.execute("""
                        INSERT INTO feeding_history (timestamp, should_feed, amount_grams, growth_stage, fish_count, hunger_status, water_quality_status, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (past_time, 1, dosage, stage, count, "Hungry", "Optimal", f"Scheduled Feed ({stage} count: {count})"))

            # Log system event
            cursor.execute("""
                INSERT INTO system_logs (timestamp, level, module, message)
                VALUES (datetime('now', 'localtime'), 'INFO', 'DatabaseManager', 'Seeded demonstration records successfully.')
            """)
            conn.commit()
            logger.info("Demo data seeded successfully into SQLite database.")

    def clear_tables(self) -> None:
        """Clear all historical records from tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for tbl in ["fish_detection", "water_quality", "predictions", "feeding_history", "system_logs"]:
                cursor.execute(f"DELETE FROM {tbl}")
            conn.commit()
            logger.info("Cleared all SQLite database tables.")

    def get_table_row_counts(self) -> Dict[str, int]:
        """Return row count for every managed table.

        Returns:
            Dict mapping table name → integer row count.
        """
        tables = ["fish_detection", "water_quality", "predictions",
                  "feeding_history", "system_logs"]
        counts: Dict[str, int] = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for tbl in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                counts[tbl] = cursor.fetchone()[0]
        return counts


# ===========================================================================
# Standalone Demo — Module 9
# ===========================================================================

if __name__ == "__main__":
    import logging
    import tempfile
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 65)
    print("  MODULE 9 — SQLite Database (Standalone Demo)")
    print("=" * 65)

    # Use a temp DB so the demo doesn't pollute the production DB
    tmp_db = os.path.join(tempfile.gettempdir(), "fish_ai_demo.db")
    db = DatabaseManager(db_path=tmp_db)

    # --- 1. Schema init ---
    print("\n  [1] Tables created")
    counts = db.get_table_row_counts()
    for tbl, cnt in counts.items():
        print(f"      {tbl:<30} {cnt} rows")

    # --- 2. Seed demo data ---
    print("\n  [2] Seeding 40 historical records …")
    db.seed_demo_data(num_records=40)
    counts = db.get_table_row_counts()
    for tbl, cnt in counts.items():
        print(f"      {tbl:<30} {cnt} rows")

    # --- 3. Manual logging ---
    print("\n  [3] Logging individual events …")
    db.log_detection(fish_count=9, growth_stage="Fingerling",
                     avg_conf=0.91, avg_speed=78.5, zone_count=4)
    db.log_water_quality(ph=7.3, temp=26.8, turbidity=4.1,
                         do=7.6, is_safe=True, stress_index=0.04, alerts="")
    db.log_prediction(status="Hungry", conf=0.95, score=88.0,
                      engine="RandomForest Classifier (ML)",
                      suppressed=False, factors="Speed: 78.5 px/s; Zone: 4/9")
    db.log_feeding_decision(should_feed=True, amount=3.15, stage="Fingerling",
                            count=9, hunger_status="Hungry",
                            wq_status="Optimal",
                            reason="FEEDING APPROVED: Water Safe & Fish Hungry")
    db.log_system_event("INFO", "Module9Demo", "Standalone demo completed successfully.")
    print("      ✓ Detection, Water Quality, Prediction, Feeding, System Log — all written")

    # --- 4. Summary stats ---
    print("\n  [4] Summary Statistics")
    stats = db.get_summary_stats()
    for k, v in stats.items():
        print(f"      {k:<30} {v}")

    # --- 5. Recent feedings ---
    print("\n  [5] 3 Most Recent Feeding Events")
    feedings = db.get_recent_feedings(limit=3)
    for f in feedings:
        feed_icon = "✓" if f["should_feed"] else "✗"
        print(f"      {feed_icon} {f['timestamp']}  {f['amount_grams']}g  "
              f"{f['growth_stage']}  {f['hunger_status']}")

    # --- 6. Cleanup ---
    db.clear_tables()
    final_counts = db.get_table_row_counts()
    print(f"\n  [6] Tables cleared — total rows: {sum(final_counts.values())}")
    os.remove(tmp_db)

    print(f"\n{'='*65}")
    print("  MODULE 9 DEMO COMPLETE — SQLite database fully operational.")
    print(f"{'='*65}")
