'''
This script fetches traffic event data from the AZ511 API and stores it in a local SQLite database.
It creates the database and table if they do not exist, and updates existing records based on event IDs.
It uses environment variables for sensitive information like the API key.
'''
import os
import requests
import sqlite3
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

def create_database():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Since az511.py is now in the database directory, the db file is in the same directory
    db_path = os.path.join(script_dir, 'az511.db') # use absolute path so it can be run in scron SLURM
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create events table
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            ID TEXT PRIMARY KEY,
            Organization TEXT,
            RoadwayName TEXT,
            DirectionOfTravel TEXT,
            Description TEXT,
            Reported INTEGER,
            LastUpdated INTEGER,
            StartDate INTEGER,
            PlannedEndDate INTEGER,
            LanesAffected TEXT,
            Latitude REAL,
            Longitude REAL,
            LatitudeSecondary REAL,
            LongitudeSecondary REAL,
            EventType TEXT,
            EventSubType TEXT,
            IsFullClosure BOOLEAN,
            Severity TEXT,
            EncodedPolyline TEXT,
            Width TEXT,
            Height TEXT,
            Length TEXT,
            Weight TEXT,
            Speed TEXT,
            DetourPolyline TEXT,
            DetourInstructions TEXT,
            Recurrence TEXT,
            RecurrenceSchedules TEXT,
            Details TEXT,
            LaneCount INTEGER
        )
    ''')
    conn.commit()
    return conn

def fetch_az511_data():
    api_key = os.getenv('AZ511_API_KEY')
    url = 'https://az511.com/api/v2/get/event'
    params = {
        'key': api_key,
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def insert_events(conn, events):
    c = conn.cursor()
    inserted = 0
    for event in events:
        # Extract restriction values
        restrictions = event.get('Restrictions', {})
        
        # Convert complex objects to strings
        recurrence = json.dumps(event.get('Recurrence')) if event.get('Recurrence') else None
        recurrence_schedules = json.dumps(event.get('RecurrenceSchedules')) if event.get('RecurrenceSchedules') else None
        
        # Prepare the data tuple with proper type handling
        event_data = (
            str(event.get('ID')) if event.get('ID') else None,
            str(event.get('Organization')) if event.get('Organization') else None,
            str(event.get('RoadwayName')) if event.get('RoadwayName') else None,
            str(event.get('DirectionOfTravel')) if event.get('DirectionOfTravel') else None,
            str(event.get('Description')) if event.get('Description') else None,
            int(event.get('Reported')) if event.get('Reported') else None,
            int(event.get('LastUpdated')) if event.get('LastUpdated') else None,
            int(event.get('StartDate')) if event.get('StartDate') else None,
            int(event.get('PlannedEndDate')) if event.get('PlannedEndDate') else None,
            str(event.get('LanesAffected')) if event.get('LanesAffected') else None,
            float(event.get('Latitude')) if event.get('Latitude') else None,
            float(event.get('Longitude')) if event.get('Longitude') else None,
            float(event.get('LatitudeSecondary')) if event.get('LatitudeSecondary') else None,
            float(event.get('LongitudeSecondary')) if event.get('LongitudeSecondary') else None,
            str(event.get('EventType')) if event.get('EventType') else None,
            str(event.get('EventSubType')) if event.get('EventSubType') else None,
            1 if event.get('IsFullClosure') else 0,
            str(event.get('Severity')) if event.get('Severity') else None,
            str(event.get('EncodedPolyline')) if event.get('EncodedPolyline') else None,
            str(restrictions.get('Width')) if restrictions.get('Width') else None,
            str(restrictions.get('Height')) if restrictions.get('Height') else None,
            str(restrictions.get('Length')) if restrictions.get('Length') else None,
            str(restrictions.get('Weight')) if restrictions.get('Weight') else None,
            str(restrictions.get('Speed')) if restrictions.get('Speed') else None,
            str(event.get('DetourPolyline')) if event.get('DetourPolyline') else None,
            str(event.get('DetourInstructions')) if event.get('DetourInstructions') else None,
            recurrence,
            recurrence_schedules,
            str(event.get('Details')) if event.get('Details') else None,
            int(event.get('LaneCount')) if event.get('LaneCount') else None
        )
        
        # Insert or replace existing event, this code delete an entry and insert if ID conflicts, can be inefficient
        c.execute('''
            INSERT OR REPLACE INTO events (
                ID, Organization, RoadwayName, DirectionOfTravel, Description,
                Reported, LastUpdated, StartDate, PlannedEndDate, LanesAffected,
                Latitude, Longitude, LatitudeSecondary, LongitudeSecondary,
                EventType, EventSubType, IsFullClosure, Severity, EncodedPolyline,
                Width, Height, Length, Weight, Speed,
                DetourPolyline, DetourInstructions, Recurrence, RecurrenceSchedules,
                Details, LaneCount
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', event_data)
        if c.rowcount == 1: # The number of rows affected by the last execute() call made on the cursor c.
            inserted += 1
    conn.commit()
    # Total entries in the DB after update
    total = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updated+Inserted: {inserted}, total: {total} events in database")

def main():
    # Create or connect to database
    conn = create_database()
    
    # Fetch data from AZ511 API
    data = fetch_az511_data()
    
    if data and isinstance(data, list):
        # Insert events into database
        insert_events(conn, data)
        # print(f"Successfully updated {len(data)} events in database")
    else:
        print("No valid events data received")
    
    conn.close()

if __name__ == "__main__":
    main()