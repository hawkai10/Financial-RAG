import sqlite3
import os
from .utils import logger
from .paths import FEEDBACK_DB

def migrate_database_complete(db_path: str = None):
    db_path = db_path or str(FEEDBACK_DB)
    """Complete database migration for Phase 1 enhancements."""
    
    if not os.path.exists(db_path):
        logger.info("No existing database found. New schema will be created.")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check existing columns in query_feedback
        cursor.execute("PRAGMA table_info(query_feedback)")
        feedback_columns = [column[1] for column in cursor.fetchall()]
        
        # Check existing columns in query_cache
        cursor.execute("PRAGMA table_info(query_cache)")
        cache_columns = [column[1] for column in cursor.fetchall()]
        
        # Add missing columns to query_feedback
        feedback_new_columns = [
            ("query_strategy", "TEXT"),
            ("query_complexity_score", "REAL"),
            ("user_agent", "TEXT"),
            ("ip_address", "TEXT")
        ]
        
        for column_name, column_type in feedback_new_columns:
            if column_name not in feedback_columns:
                try:
                    cursor.execute(f"ALTER TABLE query_feedback ADD COLUMN {column_name} {column_type}")
                    logger.info(f"Added column: query_feedback.{column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.error(f"Failed to add column {column_name}: {e}")
        
        # Add missing columns to query_cache
        cache_new_columns = [
            ("cache_hit_count", "INTEGER DEFAULT 0"),
            ("strategy_used", "TEXT")
        ]
        
        for column_name, column_type in cache_new_columns:
            if column_name not in cache_columns:
                try:
                    cursor.execute(f"ALTER TABLE query_cache ADD COLUMN {column_name} {column_type}")
                    logger.info(f"Added column: query_cache.{column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.error(f"Failed to add column {column_name}: {e}")
        
        conn.commit()
        logger.info("Database migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database_complete()
