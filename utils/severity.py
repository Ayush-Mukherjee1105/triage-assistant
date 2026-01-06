from model.red_flag_predict import predict_red_flag
from model.duration_predict import predict_duration


def assess_severity(text: str, model_confidence: float):
    """
    Central triage decision function.
    ALWAYS returns exactly 4 values:
    (severity, action, reasons, severity_score)
    """

    reasons = []
    severity_score = 0

    # -------------------------
    # RED FLAG CHECK (PRIMARY)
    # -------------------------
    try:
        red_flag, rf_label, rf_conf = predict_red_flag(text)
        if red_flag:
            severity_score += 6
            reasons.append(f"Red flag detected: {rf_label}")
    except Exception:
        reasons.append("Red-flag model unavailable")

    # -------------------------
    # DURATION CHECK (SECONDARY)
    # -------------------------
    try:
        min_d, max_d = predict_duration(text)
        if max_d >= 7:
            severity_score += 2
            reasons.append(f"Symptoms may persist ({min_d}-{max_d} days)")
    except Exception:
        reasons.append("Duration model unavailable")

    # -------------------------
    # LOW CONFIDENCE ESCALATION
    # -------------------------
    if model_confidence < 0.30:
        severity_score += 1
        reasons.append("Low model confidence")

    # -------------------------
    # FINAL DECISION
    # -------------------------
    if severity_score >= 7:
        return (
            "HIGH",
            "Seek urgent medical care",
            reasons,
            severity_score
        )

    elif severity_score >= 4:
        return (
            "MODERATE",
            "Consult a doctor if symptoms persist",
            reasons,
            severity_score
        )

    else:
        return (
            "LOW",
            "Self-care and monitoring advised",
            reasons,
            severity_score
        )
