from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3
import uvicorn
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

app = FastAPI()

# Database setup
conn = sqlite3.connect("attendace.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT,
    date_time TEXT,
    similarity REAL,
    image_path TEXT
)
""")
conn.commit()

class Incident(BaseModel):
    employee_id: str
    timestamp: str
    similarity: float = None
    image_path: str

@app.post("/incident")
async def save_incident(incident: Incident):
    print(f"Received incident: {incident}")
    cursor.execute(
        "INSERT INTO incidents (employee_id, date_time, similarity, image_path) VALUES (?, ?, ?, ?)",
        (incident.employee_id, incident.timestamp, incident.similarity, incident.image_path)
    )
    conn.commit()
    return {"status": "success","message": "Attendance saved successfully"}

@app.get("/incident")
async def get_incident(employee_id: str = None, start_date: str = None, end_date: str = None):
    query = "SELECT * FROM incidents WHERE 1=1"
    params = []

    if employee_id:
        query += " AND employee_id = ?"
        params.append(employee_id)
    if start_date and end_date:
        query += " AND date(date_time) BETWEEN ? AND ?"
        params.extend([start_date, end_date])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    result = [
        {"id": row[0], "employee_id": row[1], "date_time": row[2], "similarity": row[3], "image_path": row[4]}
        for row in rows
    ]
    return {"status": "success", "records": result}

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Run the Uvicorn server with the specified port
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)