def assess_severity(
    is_red_flag: bool,
    diagnosis_confidence: float,
    duration_days: int,
    text: str,
):
    """
    Unified severity engine (research-calibrated)

    Combines:
    - red flag detection
    - symptom duration
    - diagnosis confidence
    - multilingual clinical keywords

    Returns:
        severity_label (str)
        severity_score (int)
    """

    # -------------------------------------------------
    # RULE 1 — RED FLAG OVERRIDE (MOST IMPORTANT)
    # -------------------------------------------------

    if is_red_flag:
        return "HIGH", 9

    score = 0
    text_lower = text.lower()

    # -------------------------------------------------
    # RULE 2 — DURATION SIGNAL
    # -------------------------------------------------

    # Persistent symptoms escalate severity

    if duration_days >= 30:
        score += 3

    elif duration_days >= 14:
        score += 2

    elif duration_days >= 7:
        score += 2

    elif duration_days >= 5:
        score += 1

    # -------------------------------------------------
    # RULE 3 — DIAGNOSIS CONFIDENCE SIGNAL
    # -------------------------------------------------

    if diagnosis_confidence >= 0.90:
        score += 2

    elif diagnosis_confidence >= 0.75:
        score += 1

    elif diagnosis_confidence < 0.30:
        score -= 1

    # -------------------------------------------------
    # RULE 4 — MODERATE PERSISTENCE KEYWORDS
    # -------------------------------------------------

    moderate_keywords = [

        # English
        "persistent",
        "not improving",
        "getting worse",
        "continuous",
        "for days",
        "for weeks",
        "still having",
        "since last week",

        # Hinglish
        "baar baar",
        "bar bar",
        "barbar",
        "1 week",
        "ek hafte",
        "kai din",

        # Hindi
        "कई दिन",
        "बार बार",
        "लगातार",

        # Bengalish
        "onek din",
        "koyek din",
        "bar bar",
        "1 week dhore",

        # Bengali
        "কয়েকদিন",
        "অনেকদিন",
        "বারবার",
    ]

    if any(k in text_lower for k in moderate_keywords):
        score += 1

    # -------------------------------------------------
    # RULE 5 — HIGH-RISK SYMPTOM KEYWORDS
    # -------------------------------------------------

    high_keywords = [

        # English
        "severe",
        "unbearable",
        "can't breathe",
        "breathless",
        "chest pain",
        "fainted",
        "loss of vision",
        "vomiting blood",
        "blood in vomit",
        "seizure",
        "confusion",

        # Hinglish
        "saans lene mein dikkat",
        "saans nahi aa rahi",
        "chakkar aake gir gaya",

        # Hindi
        "सांस लेने में दिक्कत",
        "बेहोश",
        "तेज दर्द",

        # Bengalish
        "shash nite oshubidha",
        "behosh hoye",
        "khub betha",

        # Bengali
        "শ্বাস নিতে কষ্ট",
        "অজ্ঞান",
        "তীব্র ব্যথা",
    ]

    if any(k in text_lower for k in high_keywords):
        score += 3

    # -------------------------------------------------
    # RULE 6 — COMMON ILLNESS KEYWORDS
    # -------------------------------------------------

    mild_keywords = [
        "cough",
        "cold",
        "headache",
        "mild pain",
        "sore throat",
        "runny nose",
        "fever",
        "vomiting",
    ]

    if any(k in text_lower for k in mild_keywords):
        score += 1

    # -------------------------------------------------
    # RULE 7 — EXTREME LONG DURATION SAFETY
    # -------------------------------------------------

    if duration_days >= 60:
        score += 2

    # -------------------------------------------------
    # CLAMP SCORE
    # -------------------------------------------------

    score = max(0, min(score, 10))

    # -------------------------------------------------
    # FINAL CALIBRATED MAPPING
    # -------------------------------------------------

    if score >= 7:
        severity = "HIGH"

    elif score >= 3:
        severity = "MODERATE"

    else:
        severity = "LOW"

    return severity, score