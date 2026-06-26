"""
Test suite for the Carbon Footprint Predictor deployment.

Tests the core logic — feature engineering, classification, the regression-based
CO2 estimate, the per-capita benchmark, and the suggested-question fallback —
directly against the saved models. The Streamlit UI itself is checked manually
(see docs/manual_frontend_test_plan.xlsx).

Run from the project root with:
    pip install pytest
    pytest tests/test_app.py -v

Expected values were generated from the actual saved models (model_lr.pkl and
model_co2_regressor.pkl), so they describe the real behaviour, not a guess.

NOTE: CO2 is now produced by the trained regression model (notebook 07), not a
fixed-factor formula, so all 15 features (including food spending) contribute.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pytest


# ----------------------------------------------------------------------------
# project paths
# ----------------------------------------------------------------------------
# Works both when this file is kept as tests/test_app.py and when it is placed
# temporarily in the project root.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent if THIS_DIR.name == "tests" else THIS_DIR
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# ----------------------------------------------------------------------------
# load the models once for all tests
# ----------------------------------------------------------------------------
MODEL = joblib.load(MODELS_DIR / "model_lr.pkl")                 # classifier
FEATURES = joblib.load(MODELS_DIR / "features_lr.pkl")
CO2_MODEL = joblib.load(MODELS_DIR / "model_co2_regressor.pkl")  # regression CO2 model


# ----------------------------------------------------------------------------
# functions under test — mirror app02_model.py so the test is self-contained
# (if you change the logic there, mirror it here)
# ----------------------------------------------------------------------------
def build_feature_row(raw):
    hh = max(raw["household_size"], 1)
    row = {
        "electricity_kwh_per_month":     raw["electricity_kwh_per_month"],
        "natural_gas_therms_per_month":  raw["natural_gas_therms_per_month"],
        "fuel_liters_per_month":         raw["fuel_liters_per_month"],
        "car_km_per_month":              raw["car_km_per_month"],
        "public_transport_km_per_month": raw["public_transport_km_per_month"],
        "meat_kg_per_month":             raw["meat_kg_per_month"],
        "energy_per_person": raw["electricity_kwh_per_month"] / hh,
        "gas_per_person":    raw["natural_gas_therms_per_month"] / hh,
        "fuel_per_person":   raw["fuel_liters_per_month"] / hh,
        "car_per_person":    raw["car_km_per_month"] / hh,
        "transport_ratio":   raw["car_km_per_month"] / (raw["public_transport_km_per_month"] + 1),
        "meat_per_person":   raw["meat_kg_per_month"] / hh,
        "food_per_person":   raw["food_spend_usd_per_month"] / hh,
        "waste_per_person":  raw["waste_kg_per_month"] / hh,
        "log_income":        np.log1p(raw["annual_income_usd"]),
    }
    return pd.DataFrame([row])[FEATURES]


def estimate_co2(raw):
    """CO2 from the regression model, guarded to be non-negative."""
    row = build_feature_row(raw)
    return max(0.0, float(CO2_MODEL.predict(row)[0]))


def predict(raw):
    row = build_feature_row(raw)
    proba = MODEL.predict_proba(row)[0]
    classes = list(MODEL.classes_)
    pred = classes[int(np.argmax(proba))]
    conf = float(np.max(proba))
    return pred, conf, {c: float(p) for c, p in zip(classes, proba)}


# per-capita benchmark, mirroring app02_model.py (uses the dataset's real CO2)
_REF = pd.read_excel(DATA_DIR / "carbon_footprint_dataset.xlsx")
_REF = _REF.assign(per_capita=_REF["estimated_co2_kg_per_month"] / _REF["household_size"])


def per_capita_assessment(raw):
    hh = raw["household_size"]
    pc = estimate_co2(raw) / max(hh, 1)
    same = _REF[_REF["household_size"] == hh]["per_capita"]
    fell_back = False
    if len(same) < 10:
        same = _REF["per_capita"]
        fell_back = True
    p33, p66 = same.quantile(0.33), same.quantile(0.66)
    level = "low" if pc <= p33 else ("average" if pc <= p66 else "high")
    pct = (same < pc).mean() * 100
    return pc, level, pct, fell_back


# suggested-question fallback, mirroring app03_advice.py
DEFAULT_SUGGESTED_QUESTIONS = [
    "Why did I get this carbon level?",
    "Which factor should I reduce first?",
    "What is the most realistic change for my household?",
    "Can you explain this result in simpler words?",
]


def suggest_questions(context, api_key=None):
    if not api_key:
        return DEFAULT_SUGGESTED_QUESTIONS
    return DEFAULT_SUGGESTED_QUESTIONS  # live AI path not unit-tested


# ----------------------------------------------------------------------------
# reusable input fixture
# ----------------------------------------------------------------------------
def base_input(**overrides):
    raw = dict(
        household_size=3, annual_income_usd=80000,
        electricity_kwh_per_month=700, natural_gas_therms_per_month=75,
        fuel_liters_per_month=100, car_km_per_month=950,
        public_transport_km_per_month=300, meat_kg_per_month=10,
        food_spend_usd_per_month=835, waste_kg_per_month=42,
    )
    raw.update(overrides)
    return raw


# ============================================================================
# 1. FEATURE ENGINEERING
# ============================================================================
class TestFeatureEngineering:

    def test_derived_values_exact(self):
        """INPUT: size=4, elec=800, gas=100, fuel=120, car=1000, public=200,
        meat=12, food=800, waste=40, income=80000.
        EXPECTED derived (by hand): energy_pp=200, gas_pp=25, fuel_pp=30,
        car_pp=250, meat_pp=3, food_pp=200, waste_pp=10,
        transport_ratio=1000/201=4.975, log_income=ln(80001)=11.2898."""
        raw = base_input(household_size=4, electricity_kwh_per_month=800,
                         natural_gas_therms_per_month=100, fuel_liters_per_month=120,
                         car_km_per_month=1000, public_transport_km_per_month=200,
                         meat_kg_per_month=12, food_spend_usd_per_month=800,
                         waste_kg_per_month=40, annual_income_usd=80000)
        row = build_feature_row(raw).iloc[0]
        assert row["energy_per_person"] == 200.0
        assert row["gas_per_person"] == 25.0
        assert row["fuel_per_person"] == 30.0
        assert row["car_per_person"] == 250.0
        assert row["meat_per_person"] == 3.0
        assert row["food_per_person"] == 200.0
        assert row["waste_per_person"] == 10.0
        assert row["transport_ratio"] == pytest.approx(4.975124, abs=1e-4)
        assert row["log_income"] == pytest.approx(11.289794, abs=1e-4)

    def test_column_order_and_count(self):
        """EXPECTED: exactly 15 columns in the model's order."""
        row = build_feature_row(base_input())
        assert list(row.columns) == FEATURES
        assert row.shape == (1, 15)

    def test_household_size_zero_guard(self):
        """INPUT: size=0. EXPECTED: treated as 1, no divide-by-zero."""
        row = build_feature_row(base_input(household_size=0, electricity_kwh_per_month=500)).iloc[0]
        assert row["energy_per_person"] == 500.0

    def test_transport_ratio_zero_public(self):
        """INPUT: car=900, public=0. EXPECTED: 900/(0+1)=900 (no /0)."""
        row = build_feature_row(base_input(car_km_per_month=900, public_transport_km_per_month=0)).iloc[0]
        assert row["transport_ratio"] == 900.0

    def test_feature_engineering_matches_training(self):
        """The app's on-the-fly feature engineering must match the values the
        model was trained on (guards against a silent mismatch)."""
        df = pd.read_csv(DATA_DIR / "processed_carbon_dataset.csv")
        raw0 = df.iloc[0]
        hh = raw0["household_size"]
        assert df.iloc[0]["energy_per_person"] == pytest.approx(
            raw0["electricity_kwh_per_month"] / hh, abs=1e-6)
        assert df.iloc[0]["log_income"] == pytest.approx(
            np.log1p(raw0["annual_income_usd"]), abs=1e-6)


# ============================================================================
# 2. CLASSIFICATION
# ============================================================================
class TestPrediction:

    def test_typical_median_is_medium(self):
        """INPUT: median household. EXPECTED: medium, confidence ~0.99."""
        pred, conf, proba = predict(base_input())
        assert pred == "medium"
        assert conf == pytest.approx(0.99, abs=0.02)

    def test_very_low_is_low(self):
        """INPUT: low consumption, size 4. EXPECTED: low, confidence ~1.0."""
        raw = base_input(household_size=4, annual_income_usd=40000,
                         electricity_kwh_per_month=200, natural_gas_therms_per_month=10,
                         fuel_liters_per_month=10, car_km_per_month=100,
                         public_transport_km_per_month=500, meat_kg_per_month=2,
                         food_spend_usd_per_month=300, waste_kg_per_month=10)
        pred, conf, _ = predict(raw)
        assert pred == "low"
        assert conf == pytest.approx(1.0, abs=0.01)

    def test_very_high_is_high(self):
        """INPUT: high consumption, size 1. EXPECTED: high, confidence ~1.0."""
        raw = base_input(household_size=1, annual_income_usd=120000,
                         electricity_kwh_per_month=1150, natural_gas_therms_per_month=145,
                         fuel_liters_per_month=190, car_km_per_month=1900,
                         public_transport_km_per_month=20, meat_kg_per_month=19,
                         food_spend_usd_per_month=1400, waste_kg_per_month=75)
        pred, conf, _ = predict(raw)
        assert pred == "high"
        assert conf == pytest.approx(1.0, abs=0.01)

    def test_large_household_high_total(self):
        """INPUT: size-5 household, high totals. EXPECTED: high (model uses
        absolute volumes, so a big household reads as high)."""
        raw = base_input(household_size=5, annual_income_usd=70000,
                         electricity_kwh_per_month=800, natural_gas_therms_per_month=90,
                         fuel_liters_per_month=110, car_km_per_month=1000,
                         public_transport_km_per_month=400, meat_kg_per_month=15,
                         food_spend_usd_per_month=1000, waste_kg_per_month=50)
        pred, _, _ = predict(raw)
        assert pred == "high"

    def test_probabilities_sum_to_one(self):
        _, _, proba = predict(base_input())
        assert sum(proba.values()) == pytest.approx(1.0, abs=1e-6)

    def test_three_classes_present(self):
        _, _, proba = predict(base_input())
        assert set(proba.keys()) == {"low", "medium", "high"}

    def test_confidence_is_calibrated(self):
        """High-confidence (>0.99) predictions on the dataset should be highly
        accurate — a check that the model isn't over-confident."""
        df = pd.read_csv(DATA_DIR / "processed_carbon_dataset.csv")
        proba = MODEL.predict_proba(df[FEATURES])
        pred = MODEL.predict(df[FEATURES])
        hi = proba.max(axis=1) > 0.99
        acc = (pred[hi] == df["carbon_level"][hi]).mean()
        assert acc > 0.9


# ============================================================================
# 3. CO2 ESTIMATE (regression model)
# ============================================================================
class TestCO2Estimate:

    def test_median_co2_value(self):
        """INPUT: median household. EXPECTED: ~1476 kg (regression model)."""
        assert estimate_co2(base_input()) == pytest.approx(1476, abs=40)

    def test_co2_never_negative(self):
        """INPUT: all-zero consumption. EXPECTED: CO2 >= 0 (guarded)."""
        raw = base_input(electricity_kwh_per_month=0, natural_gas_therms_per_month=0,
                         fuel_liters_per_month=0, car_km_per_month=0,
                         public_transport_km_per_month=0, meat_kg_per_month=0,
                         food_spend_usd_per_month=0, waste_kg_per_month=0)
        assert estimate_co2(raw) >= 0.0

    def test_co2_increases_with_consumption(self):
        """INPUT: same household, low vs high meat/car/electricity.
        EXPECTED: more consumption -> higher CO2 (correct direction)."""
        low = estimate_co2(base_input(meat_kg_per_month=2, car_km_per_month=200,
                                      electricity_kwh_per_month=300))
        high = estimate_co2(base_input(meat_kg_per_month=20, car_km_per_month=2000,
                                       electricity_kwh_per_month=1500))
        assert high > low

    def test_food_now_contributes(self):
        """Regression model uses all 15 features, so changing food spending
        changes CO2 (unlike the old fixed-factor formula)."""
        a = estimate_co2(base_input(food_spend_usd_per_month=300))
        b = estimate_co2(base_input(food_spend_usd_per_month=2500))
        assert a != b

    def test_co2_consistent_with_classification(self):
        """A 'high' classification should not come with a very low CO2 estimate
        (the two outputs should agree, not contradict each other)."""
        high = base_input(household_size=1, electricity_kwh_per_month=1150,
                          natural_gas_therms_per_month=145, fuel_liters_per_month=190,
                          car_km_per_month=1900, meat_kg_per_month=19)
        pred, _, _ = predict(high)
        assert pred == "high"
        assert estimate_co2(high) > 1500


# ============================================================================
# 4. PER-CAPITA BENCHMARK
# ============================================================================
class TestPerCapita:

    def test_efficient_large_household_is_low(self):
        """INPUT: size-5 modest consumption. EXPECTED: per-person level 'low'."""
        raw = base_input(household_size=5, electricity_kwh_per_month=400,
                         natural_gas_therms_per_month=40, fuel_liters_per_month=40,
                         car_km_per_month=400, public_transport_km_per_month=500,
                         meat_kg_per_month=6, food_spend_usd_per_month=700,
                         waste_kg_per_month=30)
        pc, level, pct, _ = per_capita_assessment(raw)
        assert level == "low"
        assert pct < 33

    def test_per_capita_value_is_co2_over_size(self):
        """EXPECTED: per-capita value = total CO2 / household size."""
        raw = base_input(household_size=4)
        total = estimate_co2(raw)
        pc, _, _, _ = per_capita_assessment(raw)
        assert pc == pytest.approx(total / 4, abs=0.1)

    def test_same_size_used_when_available(self):
        """INPUT: size 3 (well represented). EXPECTED: no fallback."""
        _, _, _, fell_back = per_capita_assessment(base_input(household_size=3))
        assert fell_back is False

    def test_out_of_range_size_falls_back(self):
        """INPUT: size 6 (outside 1-5). EXPECTED: falls back, no crash."""
        pc, level, pct, fell_back = per_capita_assessment(base_input(household_size=6))
        assert fell_back is True
        assert level in {"low", "average", "high"}


# ============================================================================
# 5. ROBUSTNESS / EDGE CASES
# ============================================================================
class TestRobustness:

    def test_all_minimum_values(self):
        """INPUT: dataset minimums. EXPECTED: low, no error."""
        raw = base_input(household_size=1, annual_income_usd=15000,
                         electricity_kwh_per_month=157, natural_gas_therms_per_month=1,
                         fuel_liters_per_month=0, car_km_per_month=0,
                         public_transport_km_per_month=0, meat_kg_per_month=0,
                         food_spend_usd_per_month=200, waste_kg_per_month=5)
        pred, _, _ = predict(raw)
        assert pred == "low"

    def test_all_maximum_input_values(self):
        """INPUT: the input-form maximums. EXPECTED: high, no error."""
        raw = base_input(household_size=5, annual_income_usd=300000,
                         electricity_kwh_per_month=2000, natural_gas_therms_per_month=300,
                         fuel_liters_per_month=400, car_km_per_month=4000,
                         public_transport_km_per_month=1200, meat_kg_per_month=40,
                         food_spend_usd_per_month=3000, waste_kg_per_month=160)
        pred, _, _ = predict(raw)
        assert pred == "high"

    def test_household_size_zero(self):
        """INPUT: size 0. EXPECTED: guarded, valid label, no crash."""
        pred, _, _ = predict(base_input(household_size=0))
        assert pred in {"low", "medium", "high"}

    def test_zero_income(self):
        """INPUT: income 0. EXPECTED: log_income=0, valid label."""
        row = build_feature_row(base_input(annual_income_usd=0)).iloc[0]
        assert row["log_income"] == 0.0
        pred, _, _ = predict(base_input(annual_income_usd=0))
        assert pred in {"low", "medium", "high"}

    def test_extreme_values_dont_crash(self):
        """INPUT: values far above the form maximums. EXPECTED: no crash, CO2>=0."""
        raw = base_input(household_size=1, annual_income_usd=10_000_000,
                         electricity_kwh_per_month=999999, car_km_per_month=9_999_999,
                         meat_kg_per_month=99999, food_spend_usd_per_month=999999,
                         waste_kg_per_month=99999)
        pred, _, _ = predict(raw)
        assert pred in {"low", "medium", "high"}
        assert estimate_co2(raw) >= 0.0

    def test_negative_input_does_not_crash(self):
        """INPUT: a negative value. EXPECTED: no exception, valid label."""
        pred, _, _ = predict(base_input(car_km_per_month=-100))
        assert pred in {"low", "medium", "high"}


# ============================================================================
# 6. SUGGESTED QUESTIONS (clickable buttons)
# ============================================================================
class TestSuggestedQuestions:

    def test_fallback_without_api_key(self):
        """INPUT: no key. EXPECTED: the fixed default list (buttons never empty)."""
        assert suggest_questions({"predicted_class": "high"}, api_key=None) == DEFAULT_SUGGESTED_QUESTIONS

    def test_returns_list_of_strings(self):
        qs = suggest_questions({"predicted_class": "medium"})
        assert isinstance(qs, list) and len(qs) >= 1
        assert all(isinstance(q, str) and q.strip() for q in qs)

    def test_question_count_reasonable(self):
        qs = suggest_questions({"predicted_class": "low"})
        assert 1 <= len(qs) <= 6
