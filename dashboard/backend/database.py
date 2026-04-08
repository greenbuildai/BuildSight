import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime

DATABASE_PATH = Path(__file__).parent / "buildsight.db"

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Detection stats for analytics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            worker_count INTEGER,
            helmet_count INTEGER,
            vest_count INTEGER,
            compliance_score FLOAT,
            unsafe_proximity_count INTEGER,
            site_condition TEXT
        )
    ''')
    
    # Alert history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            type TEXT,
            message TEXT,
            severity TEXT,
            zone TEXT
        )
    ''')
    
    # GeoAI Zones
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS geo_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            geojson TEXT,
            risk_level TEXT
        )
    ''')

    # Settings persistence
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Seed threshold if not exists
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('detection_threshold', '0.20')")

    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def log_metrics(worker_count, helmet_count, vest_count, compliance_score, unsafe_proximity=0, condition="normal"):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO metrics (worker_count, helmet_count, vest_count, compliance_score, unsafe_proximity_count, site_condition)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (worker_count, helmet_count, vest_count, compliance_score, unsafe_proximity, condition))
    conn.commit()
    conn.close()

def get_analytics_summary(days=7):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Mock aggregation for now, but querying real data
    cursor.execute('''
        SELECT 
            strftime('%Y-%m-%d', timestamp) as date,
            AVG(compliance_score) as avg_compliance,
            MAX(worker_count) as peak_workers,
            SUM(unsafe_proximity_count) as total_incidents
        FROM metrics
        WHERE timestamp >= date('now', ?)
        GROUP BY date
        ORDER BY date ASC
    ''', (f'-{days} days',))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
