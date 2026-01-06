import streamlit as st

from utils.language import detect_language
from utils.intent import detect_intent
from utils.severity import assess_severity
from knowledge_graph.kg import enrich_prediction
from model.diagnosis_predict import predict_diagnosis
from config import CONFIDENCE_THRESHOLD
from model.xai import explain

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="AI Clinical Triage Assistant",
    layout="centered"
)

st.title("AI Clinical Triage Assistant")
st.caption(
    "AI-assisted preliminary triage guidance — not a medical diagnosis."
)

text = st.text_area("Describe your symptoms")

# -------------------------
# ANALYSIS PIPELINE
# -------------------------

if st.button("Analyze") and text.strip():

    # --- Language ---
    language, lang_conf = detect_language(text)

    # --- Intent ---
    intent = detect_intent(text)

    # --- Diagnosis model (informational only) ---
    label, model_conf = predict_diagnosis(text)

    # --- Severity (PRIMARY, SAFETY-FIRST) ---
    severity, action, reasons, severity_score = assess_severity(
        text=text,
        model_confidence=model_conf
    )

    # -------------------------
    # DISPLAY RESULTS
    # -------------------------

    st.divider()
    st.subheader("AI Assessment")

    st.write(f"**Detected Language:** {language} (confidence: {lang_conf:.2f})")
    st.write(f"**Model Confidence:** `{model_conf:.2f}`")

    # -------------------------
    # SEVERITY
    # -------------------------

    st.subheader("Severity Assessment")

    if severity == "HIGH":
        st.error("!!Urgent medical attention recommended!!")
    elif severity == "MODERATE":
        st.warning("!Medical evaluation advised!")
    else:
        st.success("Self-care advised")

    st.write(f"**Severity Level:** {severity}")
    st.write(f"**Recommended Action:** {action}")
    st.write(f"**Severity Score:** {severity_score}/10")

    # -------------------------
    # INTENT CONTEXT
    # -------------------------

    if intent == "INJURY":
        st.info(
            "Injury-related symptoms detected. "
            "Watch for worsening pain, confusion, vomiting, or loss of consciousness."
        )

    elif intent == "LIFESTYLE":
        st.info(
            "Lifestyle-related concern detected. "
            "Stress, sleep disruption, or fatigue may be contributing factors."
        )

    # -------------------------
    # DIAGNOSIS + KNOWLEDGE GRAPH
    # -------------------------

    if model_conf >= CONFIDENCE_THRESHOLD:
        st.subheader("Likely Condition (Model-Assisted)")
        st.write(f"**Likely Condition:** {label}")

        guidance = enrich_prediction(label)
        st.write(guidance["description"])
        st.write(f"**Suggested Care Path:** {guidance['action']}")
    else:
        st.warning(
            "This assessment is precautionary due to low model confidence. "
            "Providing more symptom details may improve accuracy."
        )

    # -------------------------
    # EXPLANATION
    # -------------------------

    st.subheader("Model Explanation (XAI)")
    try:
        explanations = explain(text)
        for tok, score in explanations:
            st.write(f"`{tok}` → {score:.4f}")
    except Exception:
        st.write("Explanation unavailable.")
        
    st.caption(
        "⚠️ This tool does not replace professional medical advice."
    )
