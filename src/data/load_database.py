import sqlite3
import pandas as pd
from pathlib import Path
from build_database import SQL_PATH

def load_database():
    conn = sqlite3.connect(SQL_PATH)
    
    accidents = """
    SELECT
        a.severity,

        CAST(strftime('%H', a.start_time) AS INTEGER) AS hour,
        CAST(strftime('%w', a.start_time) AS INTEGER) AS day,
        CAST(strftime('%m', a.start_time) AS INTEGER) AS month,

        CASE
        WHEN CAST(strftime('%w', a.start_time) AS INTEGER) IN (0,6)
        THEN 1 ELSE 0
        END AS is_weekend,

        CASE
        WHEN CAST(strftime('%H', a.start_time) AS INTEGER) BETWEEN 20 AND 23
            OR CAST(strftime('%H', a.start_time) AS INTEGER) BETWEEN 0 AND 5
        THEN 1 ELSE 0
        END AS is_night,

        l.state,
        l.latitude,
        l.longitude,

        w.temperature_f,
        w.visibility_mi,
        w.wind_speed_mph,
        w.precipitation_in,
        w.weather_condition,

        r.junction,
        r.traffic_signal,
        r.crossing,
        r.stop,
        r.railway,
        r.roundabout,
        r.bump,

        m.amenity,
        m.give_way,
        m.no_exit,
        m.station,
        m.traffic_calming,
        m.turning_loop,

        (r.junction + r.traffic_signal + r.crossing + r.stop +
        r.railway + r.roundabout + r.bump) AS major_road_feature_count,

        (m.amenity + m.give_way + m.no_exit +
        m.station + m.traffic_calming + m.turning_loop) AS minor_road_feature_count

    FROM accidents a
    JOIN locations l
    ON a.location_id = l.location_id
    JOIN weather_conditions w
    ON a.weather_id = w.weather_id
    JOIN road_features r
    ON a.road_features_id = r.road_features_id
    JOIN minor_road_features m
    ON a.minor_road_features_id = m.minor_road_features_id
    WHERE a.severity IS NOT NULL;
    """

    df = pd.read_sql(accidents, conn)
    conn.close()

    return df

if __name__ == "__main__":
    load_database()