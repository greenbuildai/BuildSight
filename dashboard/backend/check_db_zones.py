import sqlite3
from pathlib import Path

DB_PATH = Path("E:/Company/Green Build AI/Prototypes/BuildSight/dashboard/backend/buildsight.db")

def check_zones():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM geo_zones")
    rows = cursor.fetchall()
    
    print("Zones in Database:")
    for row in rows:
        print(f" - {row['name']}")
    
    if not rows:
        print(" (No zones found in database)")
    
    conn.close()

if __name__ == "__main__":
    check_zones()
