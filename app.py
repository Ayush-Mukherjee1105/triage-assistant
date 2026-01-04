import streamlit as st

from model.predict import predict
from utils.language import detect_language
from utils.intent import detect_intent
from utils.clinical_ner import extract_clinical_entities
from utils.severity import assess_severity
from knowledge_graph.kg import enrich_prediction
from config import CONFIDENCE_THRESHOLD

st.set_page_config(
    page_title="AI Clinical Triage Assistant",
    layout="centered"
)

st.title("🩺 AI Clinical Triage Assistant")
st.caption(
    "AI-assisted preliminary triage guidance — not a medical diagnosis."
)

text = st.text_area("Describe your symptoms")

if st.button("Analyze") and text.strip():

    language, lang_conf = detect_language(text)
    intent = detect_intent(text)
    entities = extract_clinical_entities(text)

    st.divider()
    st.subheader("🧠 AI Assessment")
    st.write(f"**Detected Language:** {language} (confidence: {lang_conf:.2f})")

    label, model_conf = predict(text)
    st.write(f"**Model Confidence:** `{model_conf:.2f}`")

    severity, triage, reasons = assess_severity(
        intent=intent,
        entities=entities,
        model_confidence=model_conf
    )

    st.subheader("🚦 Triage Decision")
    st.write(f"**Severity Score:** `{severity}/10`")
    st.markdown(f"### {triage}")

    if model_conf >= CONFIDENCE_THRESHOLD:
        st.write(f"**Likely Condition:** {label}")
        guidance = enrich_prediction(label)
        st.write(guidance["description"])
        st.write(f"**Recommended Action:** {guidance['action']}")
    else:
        st.warning(
            "This assessment is precautionary due to low confidence. "
            "Providing symptom details may improve accuracy."
        )

    st.subheader("🔍 Why did the AI escalate this?")
    for r in reasons:
        st.write(f"- {r}")

    st.caption(
        "⚠️ This tool does not replace professional medical advice."
    )
