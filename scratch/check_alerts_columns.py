import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(r'E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\backend\.env')

db_config = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "buildsight_geoai"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "jovi2748"),
}

try:
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    print("\n--- Columns in alerts ---")
    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'alerts';")
    for col in cur.fetchall():
        print(f"  {col[0]}: {col[1]}")
    conn.close()
except Exception as e:
    print("FAILURE:", e)
