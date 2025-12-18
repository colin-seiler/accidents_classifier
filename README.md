https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

US_Accidents_March23.csv\
        ↓\
DuckDB (read_csv_auto + SAMPLE + projection)\
        ↓\
Pandas DataFrame\
        ↓\
SQLite: stg_accidents\
        ↓\
SQLite: locations\
SQLite: weather_conditions\
SQLite: road_features\
SQLite: minor_road_features\
        ↓\
SQLite: accidents (FACT)\
        ↓\
SQLite: modeling_view\