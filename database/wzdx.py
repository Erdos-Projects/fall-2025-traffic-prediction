'''
When you re-run wzdx.py, it fetches new work zone data from the AZ511 API and updates the work_zones table - either inserting new work zones or updating existing ones if their update_date has changed.

The daily_counts table itself is not directly updated when wzdx.py runs. Instead, it uses a "lazy update" approach:

The counts are only updated when the dashboard requests data for a date range
If there are any missing dates in the cache for the requested range, it recalculates the counts for that entire range
The counts are then stored in the daily_counts table with a last_updated timestamp
'''
import sqlite3
import requests
import json
from datetime import datetime
from pathlib import Path
import logging
import time
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample city coordinates for filtering accidents (latitude, longitude)
CITY_COORDS = {
    "Phoenix": {"lat": 33.4484, "lon": -112.0740},
    "Tucson": {"lat": 32.2226, "lon": -110.9747},
    "Mesa": {"lat": 33.4152, "lon": -111.8315},
    "Chandler": {"lat": 33.3062, "lon": -111.8413},
    "Scottsdale": {"lat": 33.4942, "lon": -111.9261},
    "Glendale": {"lat": 33.5387, "lon": -112.1850},
    "Gilbert": {"lat": 33.3528, "lon": -111.7890},
    "Tempe": {"lat": 33.4255, "lon": -111.9400},
    "Peoria": {"lat": 33.5806, "lon": -112.2374},
    "Surprise": {"lat": 33.6292, "lon": -112.3937},
    "Yuma": {"lat": 32.6927, "lon": -114.6262},
    "Flagstaff": {"lat": 35.1983, "lon": -111.6513},
    "Lake Havasu City": {"lat": 34.4839, "lon": -114.3225},
    "Kingman": {"lat": 35.1894, "lon": -113.5502},
    "Bullhead City": {"lat": 35.1201, "lon": -114.5289},
    "Casa Grande": {"lat": 32.8795, "lon": -111.7574},
    "Sierra Vista": {"lat": 31.5545, "lon": -110.3009},
    "Maricopa": {"lat": 33.0581, "lon": -112.0476},
    "Oro Valley": {"lat": 32.3912, "lon": -110.9665},
    "Drexel Heights": {"lat": 32.1663, "lon": -111.0668},
    "Green Valley": {"lat": 31.8590, "lon": -110.9938},
    "San Luis": {"lat": 32.4925, "lon": -114.8488},
    "Somerton": {"lat": 32.6654, "lon": -114.6050},
    "Wellton": {"lat": 32.6333, "lon": -113.3667},
    "Pinal County": {"lat": 32.6916, "lon": -111.4757},
    "Yavapai County": {"lat": 34.5633, "lon": -112.4685},
    "Mohave County": {"lat": 35.2140, "lon": -113.7633},
    "Coconino County": {"lat": 35.6645, "lon": -111.5357},
    "Gila County": {"lat": 33.6378, "lon": -111.1079},
    "La Paz County": {"lat": 34.1667, "lon": -113.5833},
    "Navajo County": {"lat": 35.3682, "lon": -110.5068},
    "Apache County": {"lat": 34.5333, "lon": -109.3667},
    "Santa Cruz County": {"lat": 31.5833, "lon": -110.8333},
    "Cochise County": {"lat": 31.4167, "lon": -109.9333},
    "Pima County": {"lat": 32.0662, "lon": -110.9426},
    "Maricopa County": {"lat": 33.4484, "lon": -112.0740},
    "Yuma County": {"lat": 32.6927, "lon": -114.6262},
    "Graham County": {"lat": 32.7667, "lon": -109.8833},
    "Greenlee County": {"lat": 32.0333, "lon": -109.0833},
    "Humboldt County": {"lat": 35.4000, "lon": -113.0000},
    "La Paz County": {"lat": 34.1667, "lon": -113.5833},
    "Mohave County": {"lat": 35.2140, "lon": -113.7633},
    "Navajo County": {"lat": 35.3682, "lon": -110.5068},
    "Pima County": {"lat": 32.0662, "lon": -110.9426},
    "Santa Cruz County": {"lat": 31.5833, "lon": -110.8333},
    "Yavapai County": {"lat": 34.5633, "lon": -112.4685},
}

class WorkZoneDB:
    def __init__(self, db_path="workzones.db"):
        # Create database in the same directory as the script
        self.db_path = Path(__file__).parent / db_path
        self.init_db()
        
        # Statistics for batch updates
        self.stats = {
            'updated': 0,
            'new': 0,
            'accidents': 0,
            'skipped': 0
        }

    def init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS work_zones (
                    id TEXT PRIMARY KEY,
                    event_type TEXT,
                    data_source_id TEXT,
                    road_names TEXT,
                    direction TEXT,
                    description TEXT,
                    update_date TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    road_event_id TEXT,
                    is_start_position_verified BOOLEAN,
                    is_end_position_verified BOOLEAN,
                    location_method TEXT,
                    vehicle_impact TEXT,
                    latitude REAL,
                    longitude REAL,
                    raw_data JSON,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create accidents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accidents (
                    id TEXT PRIMARY KEY,
                    work_zone_id TEXT,
                    description TEXT,
                    update_date TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    latitude REAL,
                    longitude REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(work_zone_id) REFERENCES work_zones(id)
                )
            """)
            
            # Create table for daily event counts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_counts (
                    date TEXT PRIMARY KEY,
                    event_count INTEGER,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_update_date ON work_zones(update_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON work_zones(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_location ON work_zones(latitude, longitude)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date_range ON work_zones(start_date, end_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_accidents_dates ON accidents(start_date, end_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_accidents_location ON accidents(latitude, longitude)")

    def get_existing_workzones(self) -> Dict[str, str]:
        """Get dictionary of existing work zone IDs and their update dates"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, update_date FROM work_zones")
            return dict(cursor.fetchall())

    def insert_work_zone(self, feature: dict) -> bool:
        """Insert or update a single work zone feature"""
        try:
            properties = feature['properties']
            core = properties['core_details']
            coords = feature['geometry']['coordinates'][0]
            
            # Check if this work zone needs updating
            existing_zones = self.get_existing_workzones()
            work_zone_id = feature['id']
            current_update = core.get('update_date')
            description = core.get('description', '')
            
            if work_zone_id in existing_zones:
                if existing_zones[work_zone_id] == current_update:
                    self.stats['skipped'] += 1
                    return True
                self.stats['updated'] += 1
            else:
                self.stats['new'] += 1

            with sqlite3.connect(self.db_path) as conn:
                # Insert/update work zone
                conn.execute("""
                    INSERT OR REPLACE INTO work_zones (
                        id, event_type, data_source_id, road_names,
                        direction, description, update_date,
                        start_date, end_date, road_event_id,
                        is_start_position_verified, is_end_position_verified,
                        location_method, vehicle_impact,
                        latitude, longitude, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    work_zone_id,
                    core.get('event_type'),
                    core.get('data_source_id'),
                    json.dumps(core.get('road_names', [])),
                    core.get('direction'),
                    description,
                    current_update,
                    properties.get('start_date'),
                    properties.get('end_date'),
                    properties.get('road_event_id'),
                    properties.get('is_start_position_verified'),
                    properties.get('is_end_position_verified'),
                    properties.get('location_method'),
                    properties.get('vehicle_impact'),
                    coords[1],  # latitude
                    coords[0],  # longitude
                    json.dumps(feature)
                ))
                
                # Handle accident if description contains "AccidentIncident"
                if "AccidentIncident" in description:
                    self.stats['accidents'] += 1
                    conn.execute("""
                        INSERT OR REPLACE INTO accidents (
                            id, work_zone_id, description,
                            update_date, start_date, end_date,
                            latitude, longitude
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        work_zone_id + "_accident",  # unique ID for the accident
                        work_zone_id,
                        description,
                        current_update,
                        properties.get('start_date'),
                        properties.get('end_date'),
                        coords[1],  # latitude
                        coords[0]   # longitude
                    ))
                else:
                    # Remove any existing accident entry if this is no longer an accident
                    conn.execute("DELETE FROM accidents WHERE work_zone_id = ?", (work_zone_id,))
                
                return True
        except Exception as e:
            logger.error(f"Error processing work zone {feature.get('id')}: {e}")
            return False
            
    def update_daily_counts(self, start_date: str, end_date: str):
        """Update the daily counts cache for the given date range"""
        with sqlite3.connect(self.db_path) as conn:
            # Generate date series
            conn.execute("""
                WITH RECURSIVE dates(date) AS (
                    SELECT date(?)
                    UNION ALL
                    SELECT date(date, '+1 day')
                    FROM dates
                    WHERE date < date(?)
                )
                INSERT OR REPLACE INTO daily_counts (date, event_count, last_updated)
                SELECT 
                    dates.date,
                    COUNT(DISTINCT work_zones.id),
                    CURRENT_TIMESTAMP
                FROM dates
                LEFT JOIN work_zones ON 
                    dates.date >= date(substr(work_zones.start_date, 1, 10)) AND
                    dates.date <= date(substr(work_zones.end_date, 1, 10))
                GROUP BY dates.date;
            """, (start_date, end_date))
            
    def get_daily_counts(self, start_date: str, end_date: str) -> list:
        """Get daily event counts for the given date range"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Check if we need to update the cache
            cursor = conn.execute("""
                SELECT COUNT(*) as missing
                FROM (
                    WITH RECURSIVE dates(date) AS (
                        SELECT date(?)
                        UNION ALL
                        SELECT date(date, '+1 day')
                        FROM dates
                        WHERE date < date(?)
                    )
                    SELECT dates.date
                    FROM dates
                    LEFT JOIN daily_counts ON dates.date = daily_counts.date
                    WHERE daily_counts.date IS NULL
                    OR daily_counts.last_updated < (
                        SELECT MAX(update_date) FROM work_zones
                        WHERE date(substr(start_date, 1, 10)) <= date(dates.date)
                        AND date(substr(end_date, 1, 10)) >= date(dates.date)
                    )
                )
            """, (start_date, end_date))
            
            missing_dates = cursor.fetchone()[0]
            
            if missing_dates > 0:
                self.update_daily_counts(start_date, end_date)
            
            # Return the counts
            return conn.execute("""
                SELECT date, event_count
                FROM daily_counts
                WHERE date BETWEEN date(?) AND date(?)
                ORDER BY date
            """, (start_date, end_date)).fetchall()

    def get_accidents(self, start_date: str, end_date: str, city=None) -> list:
        """Get accidents active during the selected date range and city area"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT id, description, update_date, 
                       latitude, longitude, start_date, end_date
                FROM accidents 
                WHERE update_date IS NOT NULL
                AND datetime(substr(start_date, 1, 19)) <= datetime(substr(?, 1, 19))
                AND datetime(substr(end_date, 1, 19)) >= datetime(substr(?, 1, 19))
            """
            params = [end_date, start_date]
            
            if city:
                # Add 0.1 degree radius around city center (roughly 11km)
                center = CITY_COORDS.get(city)
                if center:
                    query += """
                        AND latitude BETWEEN ? AND ?
                        AND longitude BETWEEN ? AND ?
                    """
                    params.extend([
                        center['lat'] - 0.1, center['lat'] + 0.1,
                        center['lon'] - 0.1, center['lon'] + 0.1
                    ])
            
            query += " ORDER BY datetime(substr(update_date, 1, 19)) DESC"
            return conn.execute(query, params).fetchall()

def fetch_and_store_workzones():
    """Fetch work zone data from AZ511 API and store in SQLite database"""
    api_url = "https://az511.com/api/wzdx"
    db = WorkZoneDB()
    
    try:
        logger.info("Fetching data from AZ511 API...")
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        features = data.get('features', [])
        successful = 0
        for feature in features:
            if db.insert_work_zone(feature):
                successful += 1
        
        logger.info(f"Work Zone Update Summary:")
        logger.info(f"Total work zones processed: {len(features)}")
        logger.info(f"- New entries: {db.stats['new']}")
        logger.info(f"- Updated entries: {db.stats['updated']}")
        logger.info(f"- Skipped (unchanged): {db.stats['skipped']}")
        logger.info(f"- Accidents found: {db.stats['accidents']}")
        
        return successful
    
    except Exception as e:
        logger.error(f"Error fetching/storing data: {e}")
        return 0

def run_periodic_fetch(interval_minutes: int = 5):
    """Run the fetch and store operation periodically"""
    logger.info(f"Starting periodic fetch every {interval_minutes} minutes")
    
    while True:
        try:
            fetch_and_store_workzones()
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Stopping periodic fetch")
            break
        except Exception as e:
            logger.error(f"Error in periodic fetch: {e}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    run_periodic_fetch(5)