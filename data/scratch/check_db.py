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
    print("SUCCESS: Connected to PostGIS")
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cur.fetchall()
    print("Existing tables:", [t[0] for t in tables])
    conn.close()
except Exception as e:
    print("FAILURE:", e)
