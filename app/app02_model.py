"""
app02_model.py — model loading, feature engineering, prediction, and the
data-driven per-capita benchmark.

These functions take the user's raw inputs and turn them into predictions and
CO2 estimates. They are imported by app01_main.py and app03_advice.py.
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# ----------------------------------------------------------------------------
# load model + feature list (cached so it loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODELS_DIR / "model_lr.pkl")          # full pipeline (scaler + LR)
        features = joblib.load(MODELS_DIR / "features_lr.pkl")     # 15-feature order
        return model, features
    except FileNotFoundError:
        st.error("Model files (model_lr.pkl / features_lr.pkl) were not found. "
                 "Make sure they are in the models/ folder.")
        st.stop()
    except Exception as e:
        st.error(f"The model could not be loaded ({e}). Please try again later.")
        st.stop()


@st.cache_resource
def load_background():
    # background sample for the SHAP explainer (uses the processed dataset)
    df = pd.read_csv(DATA_DIR / "processed_carbon_dataset.csv")
    _, features = load_model()
    sample_size = min(100, len(df))
    return df[features].sample(sample_size, random_state=42)


@st.cache_resource
def load_co2_regressor():
    """Load the trained regression model that predicts CO2 directly (saved from
    notebook 07). Used instead of a fixed-factor formula so all 15 features —
    including food spending — contribute to the CO2 estimate."""
    try:
        return joblib.load(MODELS_DIR / "model_co2_regressor.pkl")
    except FileNotFoundError:
        st.error("CO2 regression model (model_co2_regressor.pkl) was not found. "
                 "Make sure it is in the models/ folder.")
        st.stop()
    except Exception as e:
        st.error(f"The CO2 model could not be loaded ({e}). Please try again later.")
        st.stop()


model, FEATURES = load_model()
co2_model = load_co2_regressor()


# ----------------------------------------------------------------------------
# defaults / labels for the input form
# ----------------------------------------------------------------------------
RAW_DEFAULTS = {
    "household_size": 3,
    "annual_income_usd": 80000,
    "electricity_kwh_per_month": 700,
    "natural_gas_therms_per_month": 75,
    "fuel_liters_per_month": 100,
    "car_km_per_month": 950,
    "public_transport_km_per_month": 300,
    "meat_kg_per_month": 10,
    "food_spend_usd_per_month": 835,
    "waste_kg_per_month": 42,
}

RAW_LABELS = {
    "household_size": "Household size",
    "annual_income_usd": "Annual income",
    "electricity_kwh_per_month": "Electricity",
    "natural_gas_therms_per_month": "Natural gas",
    "fuel_liters_per_month": "Vehicle fuel",
    "car_km_per_month": "Car distance",
    "public_transport_km_per_month": "Public transport",
    "meat_kg_per_month": "Meat",
    "food_spend_usd_per_month": "Food spending",
    "waste_kg_per_month": "Waste",
}

MAX_MISSING_ALLOWED = 4

# colours for the carbon-level cards (kept here too for predict() callers)
LABEL_COLORS = {"low": "#2e7d32", "medium": "#f9a825", "high": "#c62828"}

# the consumption fields that can be filled from same-size medians
FILLABLE_FIELDS = [
    "annual_income_usd", "electricity_kwh_per_month", "natural_gas_therms_per_month",
    "fuel_liters_per_month", "car_km_per_month", "public_transport_km_per_month",
    "meat_kg_per_month", "food_spend_usd_per_month", "waste_kg_per_month",
]


@st.cache_resource
def median_tables():
    """Pre-compute, from the dataset:
      - by_size: median of each input field within each household size
      - per_capita: median of each field DIVIDED by household size (per person)
    The per-capita medians let us fill a size group that is too small/unknown by
    scaling to the user's household size (per-person median x people)."""
    df = pd.read_excel(DATA_DIR / "carbon_footprint_dataset.xlsx")
    by_size = {}
    for size, grp in df.groupby("household_size"):
        by_size[int(size)] = {f: float(grp[f].median()) for f in FILLABLE_FIELDS}
    per_capita = {f: float((df[f] / df["household_size"]).median()) for f in FILLABLE_FIELDS}
    return by_size, per_capita


def fill_missing_raw(raw):
    """Fill any missing inputs.

    - If the household size exists in the dataset: use the median of that field
      among households of the SAME size.
    - Otherwise (size too small or out of range): use the per-person median of
      that field, scaled by the user's household size (per-person median x people),
      so the filled value matches the household's size rather than a mixed-size
      overall median.

    Returns (filled_dict, fill_info) where fill_info describes each fill:
    {field, value, basis}. household_size is required and is never filled here.
    """
    by_size, per_capita = median_tables()
    filled = dict(raw)
    fill_info = []

    hh = raw.get("household_size")
    hh = int(hh) if (hh is not None and not pd.isna(hh)) else None
    size_medians = by_size.get(hh) if hh is not None else None

    for key in FILLABLE_FIELDS:
        value = raw.get(key)
        if value is None or pd.isna(value):
            if size_medians is not None:
                filled[key] = round(size_medians[key])
                basis = f"median for {hh}-person households"
            elif hh is not None:
                # scale the per-person median up to this household's size
                filled[key] = round(per_capita[key] * hh)
                basis = f"per-person median x {hh} people"
            else:
                # no household size at all — fall back to a 1-person estimate
                filled[key] = round(per_capita[key])
                basis = "per-person median (household size unknown)"
            fill_info.append({"field": key, "value": filled[key], "basis": basis})
        else:
            filled[key] = value

    return filled, fill_info


# ----------------------------------------------------------------------------
# feature engineering: raw 10 inputs -> full 15-feature row
# ----------------------------------------------------------------------------
def build_feature_row(raw):
    """raw is a dict of the 10 user-entered values. Returns a 1-row DataFrame
    with all 15 model features in the correct order."""
    hh = max(raw["household_size"], 1)  # guard against divide-by-zero
    row = {
        "electricity_kwh_per_month":    raw["electricity_kwh_per_month"],
        "natural_gas_therms_per_month": raw["natural_gas_therms_per_month"],
        "fuel_liters_per_month":        raw["fuel_liters_per_month"],
        "car_km_per_month":             raw["car_km_per_month"],
        "public_transport_km_per_month":raw["public_transport_km_per_month"],
        "meat_kg_per_month":            raw["meat_kg_per_month"],
        # derived per-capita features
        "energy_per_person": raw["electricity_kwh_per_month"] / hh,
        "gas_per_person":    raw["natural_gas_therms_per_month"] / hh,
        "fuel_per_person":   raw["fuel_liters_per_month"] / hh,
        "car_per_person":    raw["car_km_per_month"] / hh,
        # derived ratios
        "transport_ratio":   raw["car_km_per_month"] / (raw["public_transport_km_per_month"] + 1),
        "meat_per_person":   raw["meat_kg_per_month"] / hh,
        "food_per_person":   raw["food_spend_usd_per_month"] / hh,
        "waste_per_person":  raw["waste_kg_per_month"] / hh,
        "log_income":        np.log1p(raw["annual_income_usd"]),
    }
    return pd.DataFrame([row])[FEATURES]


# ----------------------------------------------------------------------------
# CO2 estimate — predicted by the trained regression model (notebook 07), so
# all 15 features (including food spending) contribute. This replaces the old
# fixed-emission-factor formula and keeps the CO2 value consistent with the
# model-based pipeline.
# ----------------------------------------------------------------------------
def estimate_co2(raw):
    row = build_feature_row(raw)            # 15 features
    value = float(co2_model.predict(row)[0])
    return max(0.0, value)                  # guard: never show a negative CO2


# ----------------------------------------------------------------------------
# per-capita benchmark: compare a household to OTHER households of the SAME size
# (data-driven — uses the dataset distribution, not the model)
# ----------------------------------------------------------------------------
@st.cache_resource
def per_capita_reference():
    """Pre-compute per-capita CO2 per household size from the dataset's actual
    CO2 values (the same target the regression model was trained on), so a user
    can be compared against households of the same size on a consistent basis."""
    raw_df = pd.read_excel(DATA_DIR / "carbon_footprint_dataset.xlsx")
    raw_df = raw_df.assign(
        per_capita=raw_df["estimated_co2_kg_per_month"] / raw_df["household_size"])
    return raw_df[["household_size", "per_capita"]]


def per_capita_assessment(raw):
    """Return (per_capita_value, level, group_text, percentile) by comparing the
    household to others of the same size in the dataset."""
    ref = per_capita_reference()
    hh = raw["household_size"]
    pc = estimate_co2(raw) / max(hh, 1)

    same = ref[ref["household_size"] == hh]["per_capita"]
    if len(same) >= 10:
        group_text = f"households of the same size ({hh} people)"
    else:
        same = ref["per_capita"]
        group_text = "all households (no same-size group available)"

    p33, p66 = same.quantile(0.33), same.quantile(0.66)
    if pc <= p33:
        level = "low"
    elif pc <= p66:
        level = "average"
    else:
        level = "high"
    pct = (same < pc).mean() * 100
    return pc, level, group_text, pct


# ----------------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------------
def predict(raw):
    row = build_feature_row(raw)
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    pred = classes[int(np.argmax(proba))]
    conf = float(np.max(proba))
    proba_dict = {c: float(p) for c, p in zip(classes, proba)}
    return row, pred, conf, proba_dict
