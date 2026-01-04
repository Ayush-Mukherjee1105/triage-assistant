def assess_severity(intent, entities, model_confidence):
    """
    Returns:
    severity_score (0–10)
    triage_label (str)
    reasons (list[str])
    """

    severity = 0
    reasons = []

    # -------------------------
    # BASE RISK BY INTENT
    # -------------------------
    if intent == "INJURY":
        severity += 4
        reasons.append("Physical injury reported")

    elif intent == "DISEASE":
        severity += 3
        reasons.append("Possible medical condition")

    elif intent == "LIFESTYLE":
        severity += 1
        reasons.append("Lifestyle-related concern")

    # -------------------------
    # RED-FLAG ENTITIES
    # -------------------------
    if entities.get("loss_of_consciousness"):
        severity += 4
        reasons.append("Loss of consciousness")

    if entities.get("memory_loss"):
        severity += 3
        reasons.append("Memory loss or confusion")

    if entities.get("vomiting"):
        severity += 2
        reasons.append("Vomiting")

    if entities.get("fever_days", 0) >= 3:
        severity += 2
        reasons.append("Fever lasting multiple days")

    if entities.get("severe_pain"):
        severity += 2
        reasons.append("Severe pain")

    # -------------------------
    # MODEL UNCERTAINTY SAFETY
    # -------------------------
    if model_confidence < 0.2:
        severity += 1
        reasons.append("Low model confidence")

    severity = min(severity, 10)

    # -------------------------
    # TRIAGE DECISION
    # -------------------------
    if severity >= 7:
        triage = "🔴 Seek urgent medical care"
    elif severity >= 4:
        triage = "🟡 Consult a doctor"
    else:
        triage = "🟢 Self-care advised"

    return severity, triage, reasons
