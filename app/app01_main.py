"""
app01_main.py — Streamlit UI for the Carbon Footprint Predictor.

Run with:
python -m pip install -r requirements.txt
streamlit run app/app01_main.py

This file holds only the interface and page flow. The logic lives in:
  - app02_model.py   (model, feature engineering, prediction, per-capita)
  - app03_advice.py  (SHAP tips + optional Groq LLM advice/chat)
  - app04_config.py  (colours, tips text, default questions)
"""

import numpy as np
import pandas as pd
import streamlit as st

try:
    from app02_model import (
        model, FEATURES, RAW_LABELS, MAX_MISSING_ALLOWED,
        fill_missing_raw, build_feature_row, estimate_co2,
        per_capita_assessment, predict,
    )
    from app03_advice import (
        make_tips, llm_advice, llm_followup_answer, suggest_questions,
    )
    from app04_config import LABEL_COLORS, PC_COLORS, DEFAULT_SUGGESTED_QUESTIONS
except ModuleNotFoundError:
    from app.app02_model import (
        model, FEATURES, RAW_LABELS, MAX_MISSING_ALLOWED,
        fill_missing_raw, build_feature_row, estimate_co2,
        per_capita_assessment, predict,
    )
    from app.app03_advice import (
        make_tips, llm_advice, llm_followup_answer, suggest_questions,
    )
    from app.app04_config import LABEL_COLORS, PC_COLORS, DEFAULT_SUGGESTED_QUESTIONS

# page config
st.set_page_config(page_title="Carbon Footprint Predictor",
                   page_icon="🌍", layout="wide")

# sidebar navigation style
st.markdown("""
<style>
.section-anchor {
    scroll-margin-top: 90px;
}

.nav-link {
    display: block;
    padding: 10px 12px;
    margin: 6px 0;
    border-radius: 8px;
    background-color: #f0f2f6;
    color: #262730 !important;
    text-decoration: none;
    font-weight: 500;
}

.nav-link:hover {
    background-color: #dfe3ea;
    color: #000000 !important;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# UI
st.title("🌍 Household Carbon Footprint Predictor")
st.caption("Enter your household's monthly data to estimate its carbon footprint "
           "and get personalised suggestions.")

# sidebar: navigation + optional LLM toggle
with st.sidebar:
    st.header("Navigation")

    st.markdown("""
    <a class="nav-link" href="#household-data">1. Household data</a>
    <a class="nav-link" href="#result-section">2. Result</a>
    <a class="nav-link" href="#suggestions-section">3. Suggestions</a>
    <a class="nav-link" href="#what-if-section">4. What-if simulator</a>
    <a class="nav-link" href="#llm-followup-section">5. AI advice & questions</a>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.header("Settings")
    use_llm = st.checkbox("Use AI-generated advice", value=True,
                          help="On by default. If the AI can't respond (no network "
                               "or quota), the app shows the rule-based tips instead. "
                               "Untick to use the rule-based tips directly.")
    api_key = ""
    if use_llm:
        # prefer the key stored in Streamlit secrets (demo mode: user need not type it)
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            st.caption("Using the pre-configured API key.")
        else:
            # fallback: let the user paste their own key
            api_key = st.text_input("API key", type="password",
                                    help="Works with any OpenAI-compatible provider "
                                         "(OpenAI, Groq free tier, OpenRouter, etc.)")

    st.markdown("---")
    st.caption("**Model:** logistic regression (15 features). "
               "Probabilities are calibrated (Brier ≈ 0.03).")
    st.caption("**Version:** 1.0 · trained on the 500-household carbon dataset.")
    st.caption("**Note:** This is an academic demonstration tool. The estimates "
               "are approximate and not a substitute for a professional carbon "
               "audit. No personal data is stored.")

# input form: the 10 raw values
st.markdown('<div id="household-data" class="section-anchor"></div>', unsafe_allow_html=True)
st.subheader("1. Your household data")
c1, c2, c3 = st.columns(3)
with c1:
    household_size = st.number_input("Household size (people) *required", min_value=1, max_value=10, value=None, step=1)
    annual_income_usd = st.number_input("Annual income (USD)", min_value=0, max_value=300000, value=None, step=1000)
    electricity = st.number_input("Electricity (kWh / month)", min_value=0, max_value=2000, value=None, step=10)
    natural_gas = st.number_input("Natural gas (therms / month)", min_value=0, max_value=300, value=None, step=5)

with c2:
    fuel = st.number_input("Vehicle fuel (litres / month)", min_value=0, max_value=400, value=None, step=5)
    car_km = st.number_input("Car distance (km / month)", min_value=0, max_value=4000, value=None, step=50)
    public_km = st.number_input("Public transport (km / month)", min_value=0, max_value=1200, value=None, step=10)

with c3:
    meat = st.number_input("Meat (kg / month)", min_value=0, max_value=40, value=None, step=1)
    food_spend = st.number_input("Food spending (USD / month)", min_value=0, max_value=3000, value=None, step=10)
    waste = st.number_input("Waste (kg / month)", min_value=0, max_value=160, value=None, step=1)

raw = {
    "household_size": household_size,
    "annual_income_usd": annual_income_usd,
    "electricity_kwh_per_month": electricity,
    "natural_gas_therms_per_month": natural_gas,
    "fuel_liters_per_month": fuel,
    "car_km_per_month": car_km,
    "public_transport_km_per_month": public_km,
    "meat_kg_per_month": meat,
    "food_spend_usd_per_month": food_spend,
    "waste_kg_per_month": waste,
}

st.markdown("")
go = st.button("🔍 Predict my carbon footprint", type="primary", use_container_width=True)

# results
st.markdown('<div id="result-section" class="section-anchor"></div>', unsafe_allow_html=True)

# results
raw_for_model = raw
can_predict = True

if go:
    # household size is required: it drives the fill logic and every per-capita feature
    if raw.get("household_size") is None or pd.isna(raw.get("household_size")):
        st.error("Household size is required. Please enter it before predicting.")
        can_predict = False
    else:
        raw_for_model, fill_info = fill_missing_raw(raw)

        if len(fill_info) > MAX_MISSING_ALLOWED:
            st.error(
                f"Too many missing values ({len(fill_info)}). Please fill in more "
                f"data before predicting — at most {MAX_MISSING_ALLOWED} may be left blank."
            )
            can_predict = False

        elif fill_info:
            st.warning(
                f"{len(fill_info)} value(s) were missing and were estimated from "
                "households of your size, so the result is an approximation."
            )
            with st.expander("View automatically filled values"):
                filled_df = pd.DataFrame({
                    "Missing field": [RAW_LABELS[x["field"]] for x in fill_info],
                    "Value used": [x["value"] for x in fill_info],
                    "How it was estimated": [x["basis"] for x in fill_info],
                })
                st.dataframe(filled_df, use_container_width=True, hide_index=True)
                st.caption("Missing values are filled with the median for households "
                           "of the same size from the dataset. If your size group is "
                           "too small, the per-person median is scaled by your "
                           "household size instead. These are estimates, not your "
                           "real data.")

if go and can_predict:
    try:
        row, pred, conf, proba = predict(raw_for_model)
        co2 = estimate_co2(raw_for_model)
    except Exception as e:
        st.error("Something went wrong while generating the prediction. "
                 "Please check your inputs and try again.")
        st.caption(f"Technical detail: {e}")
        st.stop()

    st.subheader("2. Result")

    # out-of-distribution warning: the model was trained on households of size
    # 1-5. Larger households are an extrapolation, so flag the lower reliability.
    if raw_for_model["household_size"] > 5:
        st.warning("⚠️ The model was trained on households of 1–5 people. Your "
                   "household is larger than that range, so this prediction is an "
                   "extrapolation and should be treated as a rough indication only.")

    r1, r2 = st.columns([1, 1])
    with r1:
        color = LABEL_COLORS[pred]
        st.markdown(
            f"<div style='padding:18px;border-radius:10px;background:{color};"
            f"color:white;text-align:center'>"
            f"<div style='font-size:14px;opacity:0.9'>Carbon level</div>"
            f"<div style='font-size:34px;font-weight:700'>{pred.upper()}</div>"
            f"<div style='font-size:14px'>confidence {conf*100:.0f}%</div></div>",
            unsafe_allow_html=True,
        )
        # near-boundary warning when the model isn't confident
        if conf < 0.55:
            st.warning("⚠️ This household is near the boundary between classes — "
                       "the prediction is less certain.")
    with r2:
        st.metric("Estimated CO₂ (total)", f"{co2:,.0f} kg / month",
                  help="Direct estimate from your consumption values.")
        st.caption(f"≈ {co2*12/1000:.1f} tonnes CO₂ per year")

    # ---- per-capita view: fairer comparison against same-size households ----
    pc, pc_level, group_text, pct = per_capita_assessment(raw_for_model)
    pc_color = PC_COLORS[pc_level]
    # phrase the comparison toward the helpful direction:
    # low / average -> "lower than X%" (reassuring); high -> "higher than X%"
    if pc_level == "high":
        higher_pct = pct
        compare_text = (
            "higher than almost all comparable households"
            if higher_pct >= 99 else
            f"higher than {higher_pct:.0f}% of comparable households"
        )
    else:
        lower_pct = 100 - pct
        compare_text = (
            "lower than almost all comparable households"
            if lower_pct >= 99 else
            f"lower than {lower_pct:.0f}% of comparable households"
        )
    st.markdown(
        f"<div style='margin-top:6px;padding:14px;border-radius:10px;"
        f"border:2px solid {pc_color}'>"
        f"<div style='font-size:14px;color:#555'>Per-person emissions "
        f"(vs {group_text})</div>"
        f"<div style='font-size:24px;font-weight:700;color:{pc_color}'>"
        f"{pc:,.0f} kg / person / month — {pc_level.upper()}</div>"
        f"<div style='font-size:13px;color:#555'>{compare_text}</div></div>",
        unsafe_allow_html=True,
    )
    # explain when total and per-person levels disagree (the fairness point)
    if pred == "high" and pc_level == "low":
        st.info("Your **total** emissions are high mainly because of household "
                "size — but **per person** your household is efficient. The total "
                "reflects overall contribution; the per-person view reflects "
                "lifestyle efficiency.")
    elif pred == "low" and pc_level == "high":
        st.info("Your **total** emissions are low, but **per person** they are "
                "high for your household size — there may still be room to improve "
                "individual habits.")


    # personalised advice
    st.markdown('<div id="suggestions-section" class="section-anchor"></div>', unsafe_allow_html=True)
    st.subheader("3. Personalised suggestions")
    try:
        drivers, tips = make_tips(row, pred)
    except Exception:
        # if SHAP fails, degrade gracefully with no drivers / no tips
        drivers, tips = [], []
        st.caption("Detailed driver analysis is unavailable right now.")
    if pred == "low":
        st.success("Your footprint is already low — keep it up! "
                   "The biggest contributors below are still worth watching.")

    # SHAP drivers are always shown — they are precise model facts, not advice text
    if drivers:
        st.write("**What's driving your footprint most:**")
        for fname, val in drivers:
            st.write(f"- {fname.replace('_', ' ')}")

    # save latest prediction context for LLM follow-up chat
    st.session_state["llm_context"] = {
        "raw": raw_for_model,
        "predicted_class": pred,
        "confidence": conf,
        "co2": co2,
        "proba": proba,
        "drivers": [fname.replace("_", " ") for fname, val in drivers],
        "tips": tips,
    }
    st.session_state["llm_chat_messages"] = []  # reset chat after a new prediction

    # advice text: AI if enabled and it responds, otherwise fall back to rules.
    # When AI is on, the advice becomes the FIRST assistant message in the chat
    # box below (not shown separately here, to avoid duplication).
    ai = llm_advice(raw_for_model, pred, drivers, api_key=api_key) if use_llm else None
    if ai:
        st.session_state["llm_chat_messages"].append(
            {"role": "assistant", "content": ai})
        st.caption("See the AI advice and ask follow-up questions in the chat box below ⬇")
    else:
        if use_llm:
            st.caption("AI advice is unavailable right now — showing the standard tips instead.")
        for t in tips:
            st.info(t)

    st.session_state["last_raw"] = raw_for_model  # for the what-if simulator below
    # sync what-if sliders to the freshly entered values (only on a new prediction)
    for k in ["car_km_per_month", "meat_kg_per_month", "electricity_kwh_per_month",
              "natural_gas_therms_per_month", "fuel_liters_per_month",
              "public_transport_km_per_month", "food_spend_usd_per_month",
              "waste_kg_per_month"]:
        st.session_state[f"wif_{k}"] = int(raw_for_model[k])

# what-if simulator
st.markdown("---")
st.markdown('<div id="what-if-section" class="section-anchor"></div>', unsafe_allow_html=True)
st.subheader("4. What-if simulator")

# the simulator needs a valid baseline prediction first. If the user hasn't
# made one yet (so household size etc. may be missing), prompt them and stop —
# this also prevents a crash from a missing household_size.
base = st.session_state.get("last_raw")
if not base or base.get("household_size") is None:
    st.info("Make a prediction above first — then you can explore what-if changes here.")
    st.stop()

st.caption("Drag the sliders to see how changes would shift the prediction in real time.")

# initialise slider values once (kept in session_state so reruns don't reset
# them and steal focus / scroll the page)
for k, default in [("car_km_per_month", base["car_km_per_month"]),
                   ("meat_kg_per_month", base["meat_kg_per_month"]),
                   ("electricity_kwh_per_month", base["electricity_kwh_per_month"]),
                   ("natural_gas_therms_per_month", base["natural_gas_therms_per_month"]),
                   ("fuel_liters_per_month", base["fuel_liters_per_month"]),
                   ("public_transport_km_per_month", base["public_transport_km_per_month"]),
                   ("food_spend_usd_per_month", base["food_spend_usd_per_month"]),
                   ("waste_kg_per_month", base["waste_kg_per_month"])]:
    st.session_state.setdefault(f"wif_{k}", int(default))

s1, s2 = st.columns(2)
with s1:
    wif_car = st.slider("Car distance (km / month)", 0, 4000,
                        step=50, key="wif_car_km_per_month")
    wif_meat = st.slider("Meat (kg / month)", 0, 40,
                         step=1, key="wif_meat_kg_per_month")
    wif_elec = st.slider("Electricity (kWh / month)", 0, 2000,
                         step=10, key="wif_electricity_kwh_per_month")
    wif_food = st.slider("Food spending (USD / month)", 0, 3000,
                         step=10, key="wif_food_spend_usd_per_month")
with s2:
    wif_gas = st.slider("Natural gas (therms / month)", 0, 300,
                        step=5, key="wif_natural_gas_therms_per_month")
    wif_fuel = st.slider("Vehicle fuel (litres / month)", 0, 400,
                         step=5, key="wif_fuel_liters_per_month")
    wif_public = st.slider("Public transport (km / month)", 0, 1200,
                           step=10, key="wif_public_transport_km_per_month")
    wif_waste = st.slider("Waste (kg / month)", 0, 160,
                          step=1, key="wif_waste_kg_per_month")

wif_raw = dict(base)
wif_raw.update({
    "car_km_per_month": wif_car,
    "meat_kg_per_month": wif_meat,
    "electricity_kwh_per_month": wif_elec,
    "natural_gas_therms_per_month": wif_gas,
    "fuel_liters_per_month": wif_fuel,
    "public_transport_km_per_month": wif_public,
    "food_spend_usd_per_month": wif_food,
    "waste_kg_per_month": wif_waste,
})

_, wif_pred, wif_conf, wif_proba = predict(wif_raw)
wif_co2 = estimate_co2(wif_raw)

w1, w2 = st.columns([1, 1])
with w1:
    color = LABEL_COLORS[wif_pred]
    st.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{color};"
        f"color:white;text-align:center'>"
        f"<div style='font-size:13px;opacity:0.9'>Simulated level</div>"
        f"<div style='font-size:28px;font-weight:700'>{wif_pred.upper()}</div>"
        f"<div style='font-size:13px'>confidence {wif_conf*100:.0f}%</div></div>",
        unsafe_allow_html=True,
    )
with w2:
    base_co2 = estimate_co2(base)
    delta = wif_co2 - base_co2
    st.metric("Simulated CO₂", f"{wif_co2:,.0f} kg / month",
              delta=f"{delta:,.0f} vs current", delta_color="inverse")



# LLM follow-up chat
if use_llm and st.session_state.get("llm_context") is not None:
    st.markdown("---")
    st.markdown('<div id="llm-followup-section" class="section-anchor"></div>', unsafe_allow_html=True)
    st.subheader("5. AI advice & follow-up questions")
    st.caption("The first message is your personalised advice. Ask anything else "
               "below. (Powered by the Groq Llama-3.3-70B model.)")

    if "llm_chat_messages" not in st.session_state:
        st.session_state["llm_chat_messages"] = []

    # AI-suggested clickable questions (generated once per prediction, cached)
    ctx_key = str(st.session_state["llm_context"])
    if st.session_state.get("suggested_q_key") != ctx_key:
        st.session_state["suggested_questions"] = suggest_questions(
            st.session_state["llm_context"], api_key=api_key)
        st.session_state["suggested_q_key"] = ctx_key

    # helper: run one question through the model and append to the chat
    def ask_question(question):
        st.session_state["llm_chat_messages"].append({"role": "user", "content": question})
        answer = llm_followup_answer(
            question=question,
            context=st.session_state["llm_context"],
            chat_history=st.session_state["llm_chat_messages"][:-1],
            api_key=api_key,
        )
        st.session_state["llm_chat_messages"].append({"role": "assistant", "content": answer})

    # scrollable conversation box — first message is the AI advice
    chat_box = st.container(height=380)
    with chat_box:
        for msg in st.session_state["llm_chat_messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # input row BELOW the chat box (always visible, no need to scroll the box)
    with st.form("chat_form", clear_on_submit=True):
        c_in, c_btn = st.columns([5, 1])
        typed = c_in.text_input("Your question", label_visibility="collapsed",
                                placeholder="Type your own question…")
        sent = c_btn.form_submit_button("Send", use_container_width=True)
    if sent and typed.strip():
        ask_question(typed.strip())
        st.rerun()

    # suggested questions as buttons, under the input row
    st.write("**Suggested questions** (tap to ask):")
    sq = st.session_state.get("suggested_questions", DEFAULT_SUGGESTED_QUESTIONS)
    cols = st.columns(2)
    for i, q in enumerate(sq):
        if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
            ask_question(q)
            st.rerun()

