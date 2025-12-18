import json
import sqlite3
import pandas as pd
from pathlib import Path

from src.data.load_database import load_database

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "accidents.db"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "data_schema.json"

NUMERICAL_COLS = [
    "hour", "month", "latitude", "longitude",
    "temperature_f", "visibility_mi",
    "wind_speed_mph", "precipitation_in",
]

CATEGORICAL_COLS = [
    "state",
    "weather_condition",
]

BINARY_COLS = [
    "is_weekend", "is_night",
    "junction", "traffic_signal", "crossing",
    "stop", "railway", "roundabout", "bump",
]


def build_numerical_schema(df):
    schema = {}
    for col in NUMERICAL_COLS:
        series = df[col].dropna()
        schema[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
        }
    return schema


def build_categorical_schema(df):
    schema = {}
    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts(dropna=True)
        schema[col] = {
            "unique_values": counts.index.tolist(),
            "value_counts": counts.to_dict(),
        }
    return schema


def build_binary_schema(df):
    schema = {}
    for col in BINARY_COLS:
        counts = df[col].value_counts(dropna=True)
        schema[col] = {
            "unique_values": sorted(counts.index.tolist()),
            "value_counts": {str(k): int(v) for k, v in counts.items()},
        }
    return schema

def main():
    print("📥 Loading data from SQLite...")
    df = load_database()

    schema = {
        "numerical": build_numerical_schema(df),
        "categorical": build_categorical_schema(df),
        "binary": build_binary_schema(df),
    }

    print(f"💾 Writing schema to {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    print("🎉 Schema successfully generated!")


if __name__ == "__main__":
    main()