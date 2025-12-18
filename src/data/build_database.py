import duckdb
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "US_Accidents_March23.csv"
SQL_PATH = DATA_DIR / "accidents.db"

def load_data():
    duck = duckdb.connect()
    df = duck.execute(f"""
        SELECT
            ID AS accident_id,
            Severity,
            Start_Time,
            End_Time,
            State,
            County,
            City,
            Start_Lat,
            Start_Lng,
            "Temperature(F)" AS temperature_f,
            "Visibility(mi)" AS visibility_mi,
            "Wind_Speed(mph)" AS wind_speed_mph,
            "Precipitation(in)" AS precipitation_in,
            Weather_Condition,
            Junction,
            Traffic_Signal,
            Crossing,
            Stop,
            Railway,
            Roundabout,
            Bump,
            Amenity,
            Give_Way,
            No_Exit,
            Station,
            Traffic_Calming,
            Turning_Loop,
            Description
        FROM read_csv_auto('{CSV_PATH}')
        USING SAMPLE 2%;
    """).df()

    duck.close()

    return df

def create_stage(conn):
    df = load_data()

    df.columns = df.columns.str.lower()

    df.to_sql(
        "stg_accidents",
        conn,
        if_exists="replace",
        index=False
    )

def create_tables(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS accidents (
        accident_id INTEGER PRIMARY KEY AUTOINCREMENT,
        severity INTEGER,
        start_time TEXT,
        end_time TEXT,
        location_id INTEGER,
        weather_id INTEGER,
        road_features_id INTEGER,
        minor_road_features_id INTEGER,
        description TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS locations (
        location_id INTEGER PRIMARY KEY AUTOINCREMENT,
        state TEXT,
        county TEXT,
        city TEXT,
        latitude REAL,
        longitude REAL,
        UNIQUE(state, county, city, latitude, longitude)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS weather_conditions (
        weather_id INTEGER PRIMARY KEY AUTOINCREMENT,
        temperature_f REAL,
        visibility_mi REAL,
        wind_speed_mph REAL,
        precipitation_in REAL,
        weather_condition TEXT,
        UNIQUE(
            temperature_f,
            visibility_mi,
            wind_speed_mph,
            precipitation_in,
            weather_condition
        )
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS road_features (
        road_features_id INTEGER PRIMARY KEY AUTOINCREMENT,
        junction INTEGER,
        traffic_signal INTEGER,
        crossing INTEGER,
        stop INTEGER,
        railway INTEGER,
        roundabout INTEGER,
        bump INTEGER,
        UNIQUE(
            junction,
            traffic_signal,
            crossing,
            stop,
            railway,
            roundabout,
            bump
        )
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS minor_road_features (
        minor_road_features_id INTEGER PRIMARY KEY AUTOINCREMENT,
        amenity INTEGER,
        give_way INTEGER,
        no_exit INTEGER,
        station INTEGER,
        traffic_calming INTEGER,
        turning_loop INTEGER,
        UNIQUE (
            amenity,
            give_way,
            no_exit,
            station,
            traffic_calming,
            turning_loop
        )
    );
    """)

def populate_tables(cur):
    cur.execute("""
        INSERT OR IGNORE INTO locations (
            state,
            county,
            city,
            latitude,
            longitude
        )
        SELECT DISTINCT
            state,
            county,
            city,
            start_lat,
            start_lng
        FROM stg_accidents;
                """)
    
    cur.execute("""
        INSERT OR IGNORE INTO weather_conditions (
            temperature_f,
            visibility_mi,
            wind_speed_mph,
            precipitation_in,
            weather_condition
        )
        SELECT DISTINCT
            temperature_f,
            visibility_mi,
            wind_speed_mph,
            precipitation_in,
            weather_condition
        FROM stg_accidents;
                """)
    
    cur.execute("""
        INSERT OR IGNORE INTO road_features (
            junction,
            traffic_signal,
            crossing,
            stop,
            railway,
            roundabout,
            bump
        )
        SELECT DISTINCT
            junction,
            traffic_signal,
            crossing,
            stop,
            railway,
            roundabout,
            bump
        FROM stg_accidents;
                """)
    
    cur.execute("""
        INSERT OR IGNORE INTO minor_road_features (
            amenity,
            give_way,
            no_exit,
            station,
            traffic_calming,
            turning_loop
        )
        SELECT DISTINCT
            amenity,
            give_way,
            no_exit,
            station,
            traffic_calming,
            turning_loop
        FROM stg_accidents;
                """)
    
    cur.execute("""
        INSERT INTO accidents (
            severity,
            start_time,
            end_time,
            location_id,
            weather_id,
            road_features_id,
            minor_road_features_id,
            description
        )
        SELECT
            s.severity,
            s.start_time,
            s.end_time,
            l.location_id,
            w.weather_id,
            r.road_features_id,
            m.minor_road_features_id,
            s.description
        FROM stg_accidents s

        JOIN locations l
        ON s.state = l.state
        AND s.county = l.county
        AND s.city = l.city
        AND s.start_lat = l.latitude
        AND s.start_lng = l.longitude

        JOIN weather_conditions w
        ON s.temperature_f = w.temperature_f
        AND s.visibility_mi = w.visibility_mi
        AND s.wind_speed_mph = w.wind_speed_mph
        AND s.precipitation_in = w.precipitation_in
        AND s.weather_condition = w.weather_condition

        JOIN road_features r
        ON s.junction = r.junction
        AND s.traffic_signal = r.traffic_signal
        AND s.crossing = r.crossing
        AND s.stop = r.stop
        AND s.railway = r.railway
        AND s.roundabout = r.roundabout
        AND s.bump = r.bump

        JOIN minor_road_features m
        ON s.amenity = m.amenity
        AND s.give_way = m.give_way
        AND s.no_exit = m.no_exit
        AND s.station = m.station
        AND s.traffic_calming = m.traffic_calming
        AND s.turning_loop = m.turning_loop;
        """)
    
def drop_all_tables(cur):
    cur.executescript("""
        DROP TABLE IF EXISTS accidents;
        DROP TABLE IF EXISTS minor_road_features;
        DROP TABLE IF EXISTS road_features;
        DROP TABLE IF EXISTS weather_conditions;
        DROP TABLE IF EXISTS locations;
        DROP TABLE IF EXISTS stg_accidents;
    """)

def create_3nf(full_reset=True):
    conn = sqlite3.connect(SQL_PATH)
    cur = conn.cursor()

    if full_reset:
        drop_all_tables(cur)

    create_stage(conn)
    create_tables(cur)
    populate_tables(cur)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_3nf()