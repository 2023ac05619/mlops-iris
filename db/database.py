import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
from config import DB_PATH


class DatabaseManager:
    
    def __init__(self):
        self.db_path = DB_PATH
        
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL,
                    latency REAL
                )
            ''')
            
            # New training data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS new_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    features TEXT NOT NULL,
                    target INTEGER NOT NULL,
                    used_for_training BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
        print(f"[INFO] Database initialized at {self.db_path}")
        
    def log_prediction(self, timestamp: str, input_data: list, 
                      prediction: str, confidence: float, latency: float):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (timestamp, input_data, prediction, confidence, latency)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, json.dumps(input_data), prediction, confidence, latency))
            conn.commit()
            
    def store_new_data(self, timestamp: str, features: list, target: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO new_training_data (timestamp, features, target)
                VALUES (?, ?, ?)
            ''', (timestamp, json.dumps(features), target))
            conn.commit()
            
    def count_new_samples(self) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM new_training_data 
                    WHERE used_for_training = FALSE
                ''')
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"[ERROR] Database error counting samples: {e}")
            return 0
            
    def mark_data_as_used(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE new_training_data 
                SET used_for_training = TRUE 
                WHERE used_for_training = FALSE
            ''')
            conn.commit()
            
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_predictions': 0,
            'average_db_latency': 0.0,
            'prediction_distribution': {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic stats
                cursor.execute('SELECT COUNT(*), AVG(latency) FROM predictions')
                total, avg_latency = cursor.fetchone()
                stats['total_predictions'] = total or 0
                stats['average_db_latency'] = avg_latency or 0
                
                # Prediction distribution
                cursor.execute('SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction')
                stats['prediction_distribution'] = dict(cursor.fetchall())
                
        except sqlite3.Error as e:
            print(f"[ERROR] Could not retrieve DB stats: {e}")
            
        return stats
        
    def get_prediction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        history = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
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
            print(f"[ERROR] Could not fetch prediction history: {e}")
            
        return history


