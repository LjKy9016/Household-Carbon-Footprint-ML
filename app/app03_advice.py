"""
app03_advice.py — SHAP-driven rule tips and the optional LLM features.

Contains: the SHAP explainer, rule-based tip selection, and the Groq-powered
advice / follow-up chat / suggested-question functions. The LLM functions are
optional — without an API key they return None or a fixed fallback.
"""

import numpy as np
import streamlit as st
import shap

try:
    from app02_model import model, FEATURES, load_background
    from app04_config import TIPS, DEFAULT_SUGGESTED_QUESTIONS
except ModuleNotFoundError:
    from app.app02_model import model, FEATURES, load_background
    from app.app04_config import TIPS, DEFAULT_SUGGESTED_QUESTIONS


# ----------------------------------------------------------------------------
# SHAP explainer (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_explainer():
    bg = load_background()
    scaler = model.named_steps["scaler"]
    lr = model.named_steps["model"]
    return shap.LinearExplainer(lr, scaler.transform(bg)), scaler, lr


# ----------------------------------------------------------------------------
# rule-based tips, driven by the SHAP top contributors
# ----------------------------------------------------------------------------
def make_tips(feature_row, predicted_class, top_n=3):
    """Use SHAP values for the predicted class to find what pushes the
    household toward that class, then return matching rule-based tips."""
    explainer, scaler, lr = get_explainer()
    Xs = scaler.transform(feature_row)

    sv = explainer.shap_values(Xs)
    class_idx = list(lr.classes_).index(predicted_class)

    if isinstance(sv, list):
        contribs = np.array(sv[class_idx])[0]
    else:
        sv_arr = np.array(sv)
        if sv_arr.ndim == 3:
            if sv_arr.shape[0] == 1:
                contribs = sv_arr[0, :, class_idx]
            else:
                contribs = sv_arr[class_idx, 0, :]
        elif sv_arr.ndim == 2:
            contribs = sv_arr[0]
        else:
            raise ValueError("Unexpected SHAP value shape.")

    order = np.argsort(-contribs)
    tips, drivers = [], []
    for i in order:
        fname = FEATURES[i]
        if contribs[i] <= 0:
            break
        if TIPS.get(fname):
            drivers.append((fname, contribs[i]))
            tips.append(TIPS[fname])
        if len(tips) >= top_n:
            break
    return drivers, tips


# ----------------------------------------------------------------------------
# optional LLM advice (Groq) — returns None on any failure so the caller can
# fall back to the rule-based tips
# ----------------------------------------------------------------------------
def llm_advice(raw, predicted_class, drivers, api_key=None):
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        driver_text = ", ".join(d[0].replace("_", " ") for d in drivers)
        prompt = (
            f"A household is predicted to be a {predicted_class} carbon emitter. "
            f"The main drivers are: {driver_text}. "
            f"Give 3 short, friendly, practical tips to reduce their footprint."
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # free model on Groq
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            timeout=20,  # don't let a slow network hang the app
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def format_prediction_context(context):
    """Convert the latest prediction result into text for the LLM."""
    driver_text = ", ".join(context["drivers"]) if context["drivers"] else "None"
    return (
        f"Predicted carbon level: {context['predicted_class']}\n"
        f"Prediction confidence: {context['confidence']:.2f}\n"
        f"Estimated CO2: {context['co2']:.0f} kg per month\n"
        f"Class probabilities: {context['proba']}\n"
        f"Main SHAP drivers: {driver_text}\n"
        f"Rule-based tips: {context['tips']}\n"
        f"User input values: {context['raw']}\n"
    )


def llm_followup_answer(question, context, chat_history, api_key=None):
    """Answer follow-up questions based on the latest prediction context."""
    if not api_key:
        return "LLM follow-up is unavailable because no API key was provided."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        system_prompt = (
            "You are a carbon footprint advice assistant. "
            "Use the provided prediction context to answer the user's follow-up question. "
            "Do not change the machine learning prediction. "
            "Explain that the prediction comes from the trained logistic regression model, "
            "and that SHAP is used only to identify the main contributing factors. "
            "Give practical, realistic and concise advice. "
            "If the user asks something outside the available information, say that the app does not contain enough data."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Latest prediction context:\n" + format_prediction_context(context)},
        ]
        for msg in chat_history[-6:]:  # keep only recent turns
            messages.append(msg)
        messages.append({"role": "user", "content": question})

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # free model on Groq
            messages=messages,
            max_tokens=350,
            timeout=20,  # don't let a slow network hang the app
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(LLM follow-up unavailable: {e})"


def suggest_questions(context, api_key=None):
    """Generate 3-4 short follow-up questions tailored to the prediction.
    Falls back to a fixed list if the AI is unavailable."""
    if not api_key:
        return DEFAULT_SUGGESTED_QUESTIONS
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        prompt = (
            "Based on this household's carbon footprint result, suggest 4 short, "
            "natural follow-up questions the user might want to ask. "
            "Return ONLY the questions, one per line, no numbering, each under 12 words.\n\n"
            + format_prediction_context(context)
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # free model on Groq
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            timeout=15,
        )
        text = resp.choices[0].message.content
        qs = [line.strip(" -•*0123456789.").strip()
              for line in text.splitlines() if line.strip()]
        qs = [q for q in qs if len(q) > 5][:4]
        return qs if qs else DEFAULT_SUGGESTED_QUESTIONS
    except Exception:
        return DEFAULT_SUGGESTED_QUESTIONS
