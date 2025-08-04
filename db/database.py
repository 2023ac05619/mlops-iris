import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Enhanced centralized database operations manager with model-specific tracking."""
    
    def __init__(self):
        self.db_path = DB_PATH
        
    def init_db(self):
        """Initialize database and create all required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys = ON")
                
                # Predictions table with model tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        input_data TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        confidence REAL,
                        latency REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Enhanced training data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS new_training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        features TEXT NOT NULL,
                        target INTEGER NOT NULL,
                        metadata TEXT,
                        source TEXT DEFAULT 'api',
                        used_for_training BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Model performance tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        accuracy REAL,
                        training_timestamp TEXT,
                        run_id TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # System events table for audit trail
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        event_data TEXT,
                        user_id TEXT,
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_data_timestamp ON new_training_data(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_data_used ON new_training_data(used_for_training)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_performance_name ON model_performance(model_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)')
                
                conn.commit()
                
            logger.info(f"Database initialized successfully at {self.db_path}")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def log_prediction(self, timestamp: str, input_data: list, model_name: str,
                      prediction: str, confidence: float, latency: float):
        """Log prediction to database with model information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions (timestamp, input_data, model_name, prediction, confidence, latency)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, json.dumps(input_data), model_name, prediction, confidence, latency))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Error logging prediction: {e}")
            raise
    
    def store_new_data(self, timestamp: str, features: list, target: int, 
                      metadata: Dict[str, Any] = None, source: str = "api"):
        """Store new training data with enhanced metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO new_training_data (timestamp, features, target, metadata, source)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, json.dumps(features), target, 
                     json.dumps(metadata) if metadata else None, source))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Error storing training data: {e}")
            raise
    
    def store_model_performance(self, model_name: str, accuracy: float, 
                              training_timestamp: str, run_id: str = None,
                              metadata: Dict[str, Any] = None):
        """Store model performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_performance 
                    (model_name, accuracy, training_timestamp, run_id, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (model_name, accuracy, training_timestamp, run_id, 
                     json.dumps(metadata) if metadata else None))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Error storing model performance: {e}")
            raise
    
    def log_system_event(self, event_type: str, event_data: Dict[str, Any] = None, 
                        user_id: str = None):
        """Log system events for audit trail."""
        try:
            timestamp = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_events (event_type, event_data, user_id, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (event_type, json.dumps(event_data) if event_data else None, 
                     user_id, timestamp))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Error logging system event: {e}")
    
    def count_new_samples(self) -> int:
        """Count unused training samples."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM new_training_data 
                    WHERE used_for_training = FALSE
                ''')
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Database error counting samples: {e}")
            return 0
    
    def get_total_training_samples(self) -> int:
        """Get total number of training samples."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM new_training_data')
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Database error counting total samples: {e}")
            return 0
    
    def mark_data_as_used(self):
        """Mark training data as used for training."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE new_training_data 
                    SET used_for_training = TRUE 
                    WHERE used_for_training = FALSE
                ''')
                conn.commit()
                
                # Log system event
                self.log_system_event("training_data_marked_used", {
                    "samples_marked": cursor.rowcount
                })
                
        except sqlite3.Error as e:
            logger.error(f"Error marking data as used: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        stats = {
            'total_predictions': 0,
            'average_db_latency': 0.0,
            'prediction_distribution': {},
            'predictions_by_model': {},
            'total_training_samples': 0,
            'pending_training_samples': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic prediction stats
                cursor.execute('SELECT COUNT(*), AVG(latency) FROM predictions')
                result = cursor.fetchone()
                stats['total_predictions'] = result[0] or 0
                stats['average_db_latency'] = result[1] or 0.0
                
                # Prediction distribution by class
                cursor.execute('SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction')
                stats['prediction_distribution'] = dict(cursor.fetchall())
                
                # Predictions by model
                cursor.execute('SELECT model_name, COUNT(*) FROM predictions GROUP BY model_name')
                stats['predictions_by_model'] = dict(cursor.fetchall())
                
                # Training data stats
                cursor.execute('SELECT COUNT(*) FROM new_training_data')
                stats['total_training_samples'] = cursor.fetchone()[0] or 0
                
                cursor.execute('SELECT COUNT(*) FROM new_training_data WHERE used_for_training = FALSE')
                stats['pending_training_samples'] = cursor.fetchone()[0] or 0
                
        except sqlite3.Error as e:
            logger.error(f"Could not retrieve DB stats: {e}")
            
        return stats
    
    def get_predictions_by_model(self) -> Dict[str, int]:
        """Get prediction counts for each model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT model_name, COUNT(*) FROM predictions GROUP BY model_name')
                return dict(cursor.fetchall())
        except sqlite3.Error as e:
            logger.error(f"Error getting predictions by model: {e}")
            return {}
    
    def get_average_latency_by_model(self) -> Dict[str, float]:
        """Get average prediction latency for each model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT model_name, AVG(latency) FROM predictions GROUP BY model_name')
                return {model: float(latency) for model, latency in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Error getting average latency by model: {e}")
            return {}
    
    def get_prediction_history(self, limit: int = 10, model_name: str = None) -> List[Dict[str, Any]]:
        """Get recent prediction history with optional model filtering."""
        history = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if model_name:
                    cursor.execute('''
                        SELECT * FROM predictions 
                        WHERE model_name = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (model_name, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM predictions 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                
                for row in cursor.fetchall():
                    item = dict(row)
                    item['input_data'] = json.loads(item['input_data'])
                    history.append(item)
                    
        except sqlite3.Error as e:
            logger.error(f"Could not fetch prediction history: {e}")
            
        return history
    
    def get_recent_prediction_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get prediction statistics for the last N hours."""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Recent predictions by model
                cursor.execute('''
                    SELECT model_name, COUNT(*) FROM predictions 
                    WHERE timestamp > ? 
                    GROUP BY model_name
                ''', (cutoff_time,))
                predictions_by_model = dict(cursor.fetchall())
                
                # Recent predictions by class
                cursor.execute('''
                    SELECT prediction, COUNT(*) FROM predictions 
                    WHERE timestamp > ? 
                    GROUP BY prediction
                ''', (cutoff_time,))
                predictions_by_class = dict(cursor.fetchall())
                
                # Average latency in period
                cursor.execute('''
                    SELECT AVG(latency) FROM predictions 
                    WHERE timestamp > ?
                ''', (cutoff_time,))
                avg_latency = cursor.fetchone()[0] or 0.0
                
                return {
                    "time_period_hours": hours,
                    "predictions_by_model": predictions_by_model,
                    "predictions_by_class": predictions_by_class,
                    "average_latency": avg_latency,
                    "total_predictions": sum(predictions_by_model.values())
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting recent prediction stats: {e}")
            return {}
    
    def get_training_data_insights(self) -> Dict[str, Any]:
        """Get insights about training data distribution and quality."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total samples by target class
                cursor.execute('''
                    SELECT target, COUNT(*) FROM new_training_data 
                    GROUP BY target
                ''')
                class_distribution = dict(cursor.fetchall())
                
                # Samples by source
                cursor.execute('''
                    SELECT source, COUNT(*) FROM new_training_data 
                    GROUP BY source
                ''')
                source_distribution = dict(cursor.fetchall())
                
                # Recent samples (last 7 days)
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute('''
                    SELECT COUNT(*) FROM new_training_data 
                    WHERE timestamp > ?
                ''', (week_ago,))
                recent_samples = cursor.fetchone()[0] or 0
                
                # Used vs unused samples
                cursor.execute('''
                    SELECT used_for_training, COUNT(*) FROM new_training_data 
                    GROUP BY used_for_training
                ''')
                usage_stats = dict(cursor.fetchall())
                
                return {
                    "class_distribution": class_distribution,
                    "source_distribution": source_distribution,
                    "recent_samples_7days": recent_samples,
                    "used_samples": usage_stats.get(1, 0),  # True = 1 in SQLite
                    "unused_samples": usage_stats.get(0, 0),  # False = 0 in SQLite
                    "total_samples": sum(usage_stats.values()),
                    "timestamp": datetime.now().isoformat()
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting training data insights: {e}")
            return {"error": str(e)}
    
    def get_model_performance_history(self, model_name: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get model performance history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if model_name:
                    cursor.execute('''
                        SELECT * FROM model_performance 
                        WHERE model_name = ?
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (model_name, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM model_performance 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (limit,))
                
                history = []
                for row in cursor.fetchall():
                    item = dict(row)
                    if item['metadata']:
                        item['metadata'] = json.loads(item['metadata'])
                    history.append(item)
                
                return history
                
        except sqlite3.Error as e:
            logger.error(f"Error getting model performance history: {e}")
            return []
    
    def get_system_events(self, event_type: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get system events for audit trail."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if event_type:
                    cursor.execute('''
                        SELECT * FROM system_events 
                        WHERE event_type = ?
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (event_type, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM system_events 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (limit,))
                
                events = []
                for row in cursor.fetchall():
                    item = dict(row)
                    if item['event_data']:
                        item['event_data'] = json.loads(item['event_data'])
                    events.append(item)
                
                return events
                
        except sqlite3.Error as e:
            logger.error(f"Error getting system events: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to prevent database bloat."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean old predictions (keep recent ones)
                cursor.execute('''
                    DELETE FROM predictions 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                predictions_deleted = cursor.rowcount
                
                # Clean old system events
                cursor.execute('''
                    DELETE FROM system_events 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                events_deleted = cursor.rowcount
                
                # Keep all training data - it's valuable
                # But we can clean up old model performance records
                old_performance_cutoff = (datetime.now() - timedelta(days=180)).isoformat()
                cursor.execute('''
                    DELETE FROM model_performance 
                    WHERE created_at < ?
                ''', (old_performance_cutoff,))
                performance_deleted = cursor.rowcount
                
                conn.commit()
                
                # Log cleanup event
                self.log_system_event("database_cleanup", {
                    "predictions_deleted": predictions_deleted,
                    "events_deleted": events_deleted,
                    "performance_records_deleted": performance_deleted,
                    "days_to_keep": days_to_keep
                })
                
                logger.info(f"Database cleanup completed. Deleted {predictions_deleted} predictions, "
                          f"{events_deleted} events, {performance_deleted} performance records")
                
                return {
                    "predictions_deleted": predictions_deleted,
                    "events_deleted": events_deleted,
                    "performance_deleted": performance_deleted
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error during database cleanup: {e}")
            return {"error": str(e)}
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table sizes
                table_info = {}
                tables = ['predictions', 'new_training_data', 'model_performance', 'system_events']
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    table_info[table] = cursor.fetchone()[0]
                
                # Database file size
                db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
                
                # Oldest and newest records
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM predictions')
                pred_range = cursor.fetchone()
                
                return {
                    "database_path": str(self.db_path),
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / (1024 * 1024), 2),
                    "table_counts": table_info,
                    "oldest_prediction": pred_range[0],
                    "newest_prediction": pred_range[1],
                    "total_records": sum(table_info.values()),
                    "timestamp": datetime.now().isoformat()
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    def export_training_data(self, format_type: str = "json", used_only: bool = False) -> Dict[str, Any]:
        """Export training data for analysis or backup."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT timestamp, features, target, metadata, source, used_for_training, created_at
                    FROM new_training_data
                '''
                
                if used_only:
                    query += ' WHERE used_for_training = TRUE'
                
                query += ' ORDER BY created_at DESC'
                
                cursor.execute(query)
                
                data = []
                for row in cursor.fetchall():
                    item = dict(row)
                    item['features'] = json.loads(item['features'])
                    if item['metadata']:
                        item['metadata'] = json.loads(item['metadata'])
                    data.append(item)
                
                return {
                    "format": format_type,
                    "record_count": len(data),
                    "used_only": used_only,
                    "data": data,
                    "exported_at": datetime.now().isoformat()
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error exporting training data: {e}")
            return {"error": str(e)}
    
    def backup_database(self, backup_path: str = None) -> Dict[str, Any]:
        """Create a backup of the database."""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.db_path.parent}/backup_predictions_{timestamp}.db"
            
            # Create backup using SQLite backup API
            with sqlite3.connect(self.db_path) as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)
            
            backup_size = Path(backup_path).stat().st_size
            
            # Log backup event
            self.log_system_event("database_backup", {
                "backup_path": backup_path,
                "backup_size_bytes": backup_size,
                "original_size_bytes": Path(self.db_path).stat().st_size
            })
            
            return {
                "success": True,
                "backup_path": backup_path,
                "backup_size_mb": round(backup_size / (1024 * 1024), 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return {"success": False, "error": str(e)}