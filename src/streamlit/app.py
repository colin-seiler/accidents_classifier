import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

st.set_page_config(page_title="Accident Severity Prediction", page_icon="🏎️", layout="centered")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "data" / "accident_schema.json"

# API_URL is set in docker-compose environment
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

ROAD_FEATURES = [
    "junction",
    "traffic_signal",
    "crossing",
    "stop",
    "railway",
    "roundabout",
    "bump",
]
DAYS = {
    0: "Sunday", 1: "Monday", 2: "Tuesday",
    3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"
}

@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


schema = load_schema(SCHEMA_PATH)

numerical_features = schema.get("numerical", {})
categorical_features = schema.get("categorical", {})

# Streamlit UI
st.title("🏎️ Accident Severity Prediction App")
st.write(
    f"This app sends your inputs to the FastAPI backend at **{API_BASE_URL}** for prediction."
)

st.header("Input Features")

user_input: Dict[str, Any] = {}

# Numerical Features
st.subheader("📅 Time/Location Features")

day_label = st.selectbox(
    "Day of Week",
    options=list(DAYS.keys()),
    format_func=lambda x: DAYS[x],
    index=3  # default Wednesday
)

user_input["day"] = int(day_label)

# Decide which features use sliders
SLIDER_FEATURES = {"longitude", "latitude", "temperature_f", "visibility_mi", "wind_speed_mph", "precipitation_in", "hour", "month"}

for feature_name, stats in numerical_features.items():
    min_val = float(stats.get("min"))
    max_val = float(stats.get("max"))
    mean_val = float(stats.get("mean", (min_val + max_val) / 2))
    median_val = float(stats.get("median", mean_val))

    default_val = median_val

    label = feature_name.replace("_", " ").title()
    help_text = (
        f"Min: {min_val:.2f}, Max: {max_val:.2f}, "
        f"Mean: {mean_val:.2f}, Median: {median_val:.2f}"
    )

    if feature_name in SLIDER_FEATURES:
        # Determine step size based on range and semantics
        if feature_name in {"hour", "month"}:
            step = 1.0  # age in years, int-like
            user_input[feature_name] = st.slider(
                label,
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(default_val),
                step=1,
                help=help_text,
                key=feature_name,
            )
        else:
            if feature_name in {"temperature_f", "visibility_mi", "wind_speed_mph", "precipitation_in"}:
                step = 0.1  # more granular
            else:
                # generic heuristic for latitude/longitude
                step = 0.01
            user_input[feature_name] = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=float(default_val),
            step=step,
            help=help_text,
            key=feature_name,
        )

    else:
        range_val = max_val - min_val
        if range_val > 10000:
            step = 10.0
        elif range_val > 1000:
            step = 5.0
        elif range_val > 100:
            step = 1.0
        elif range_val > 10:
            step = 0.1
        else:
            step = 0.01

        user_input[feature_name] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=float(default_val),
            step=step,
            help=help_text,
            key=feature_name,
        )

# Categorical Features
st.subheader("Categorical Features")

for feature_name, info in categorical_features.items():
    unique_values = sorted(info.get("unique_values", []))
    value_counts = info.get("value_counts", {})

    if not unique_values:
        continue

    # Default to the most common value
    if value_counts:
        default_value = max(value_counts, key=value_counts.get)
    else:
        default_value = unique_values[0]

    try:
        default_idx = unique_values.index(default_value)
    except ValueError:
        default_idx = 0

    label = feature_name.replace("_", " ").title()

    user_input[feature_name] = st.selectbox(
        label,
        options=unique_values,
        index=default_idx,
        key=feature_name,
        help=f"Distribution: {value_counts}",
    )

for i, feature in enumerate(ROAD_FEATURES):
        user_input[feature] = int(
            st.checkbox(
                feature.replace("_", " ").title(),
                value=False,
                help=f"Whether the accident location includes a {feature.replace('_', ' ')}"
            )
        )

st.markdown("---")

hour = int(user_input["hour"])
month = int(user_input["month"])

user_input["day"] = 3

user_input["is_weekend"] = 1 if user_input["day"] in {0, 6} else 0

user_input["is_night"] = 1 if (hour >= 20 or hour <= 5) else 0

# -----------------------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------------------
if st.button("🔮 Predict", type="primary"):
    payload = {"instances": [user_input]}

    with st.spinner("Calling API for prediction..."):
        try:
            resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request to API failed: {e}")
        else:
            if resp.status_code != 200:
                st.error(f"❌ API error: HTTP {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                preds = data.get("predictions", [])

                if not preds:
                    st.warning("⚠️ No predictions returned from API.")
                else:
                    pred = preds[0]
                    st.success("✅ Prediction successful!")

                    st.subheader("Prediction Result")

                    # Display prediction with nice formatting
                    if isinstance(pred, (int, float)):
                        st.metric(label="Predicted Value", value=f"{pred:,.2f}")
                    else:
                        st.metric(label="Predicted Class", value=str(pred))

                    # Show input summary in expander
                    with st.expander("📋 View Input Summary"):
                        st.json(user_input)

st.markdown("---")
st.caption(
    f"📁 Schema: `{SCHEMA_PATH}`  \n"
    f"🌐 API: `{API_BASE_URL}`"
)