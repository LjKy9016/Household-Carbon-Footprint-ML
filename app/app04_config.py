"""
app04_config.py — shared constants for the deployment app.

Colours, rule-based tips, and default questions live here so the other modules
stay focused on logic and UI.
"""

# colours for the carbon-level cards
LABEL_COLORS = {"low": "#2e7d32", "medium": "#f9a825", "high": "#c62828"}
PC_COLORS = {"low": "#2e7d32", "average": "#f9a825", "high": "#c62828"}

# rule-based reduction tips, keyed by feature name (used when SHAP flags a driver)
TIPS = {
    "electricity_kwh_per_month": "Your electricity use is a major driver. Switching to LED lighting, efficient appliances, or a green tariff would help.",
    "energy_per_person":         "Your per-person electricity use is high. Reducing standby power and heating/cooling waste would lower it.",
    "natural_gas_therms_per_month": "Natural gas is a big contributor. Better insulation or a lower thermostat setting would cut this.",
    "gas_per_person":            "Per-person gas use is high — insulation and heating habits are the main levers.",
    "fuel_liters_per_month":     "Fuel use stands out. Combining trips or shifting to public transport would reduce it.",
    "fuel_per_person":           "Per-person fuel use is high; car-sharing or fewer solo trips would help.",
    "car_km_per_month":          "Car distance is a key driver. Cycling, walking, or public transport for short trips makes a difference.",
    "car_per_person":            "Per-person car distance is high — consider car-pooling or combining journeys.",
    "transport_ratio":           "You rely heavily on the car versus public transport. Shifting some trips to transit would lower emissions.",
    "meat_kg_per_month":         "Meat consumption is a strong contributor. A few plant-based meals per week would noticeably help.",
    "meat_per_person":           "Per-person meat intake is high; reducing red meat in particular has a large effect.",
    "food_per_person":           "Food spending per person is high — reducing food waste and processed items helps.",
    "waste_per_person":          "Per-person waste is high. Recycling and composting would reduce this.",
    "log_income":                "",  # income isn't an actionable lever, skip
    "public_transport_km_per_month": "",
}

# fallback follow-up questions when the AI can't generate any
DEFAULT_SUGGESTED_QUESTIONS = [
    "Why did I get this carbon level?",
    "Which factor should I reduce first?",
    "What is the most realistic change for my household?",
    "Can you explain this result in simpler words?",
]
