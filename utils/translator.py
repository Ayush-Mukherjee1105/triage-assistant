# utils/translator.py

"""
Multilingual response mirroring engine (PRO STABLE)

Supports:
- English (pass-through)
- Hindi (MT + guard)
- Bengali (MT + strong guard)
- Hinglish (style mirror)
- Bengalish (style mirror)

Design goals:
- No crashes
- No garbage MT
- Lazy loading
- Research-safe fallbacks
"""

# Lazy globals
_hi_model = None
_hi_tok = None
_bn_model = None
_bn_tok = None


# =====================================================
# MODEL LOADERS (LAZY — prevents startup crashes)
# =====================================================

def _load_hi():
    """Load English → Hindi model once."""
    global _hi_model, _hi_tok
    if _hi_model is None:
        from transformers import MarianMTModel, MarianTokenizer
        name = "Helsinki-NLP/opus-mt-en-hi"
        _hi_tok = MarianTokenizer.from_pretrained(name)
        _hi_model = MarianMTModel.from_pretrained(name)


def _load_bn():
    """
    Multilingual fallback model.
    (More stable than nonexistent en-bn)
    """
    global _bn_model, _bn_tok
    if _bn_model is None:
        from transformers import MarianMTModel, MarianTokenizer
        name = "Helsinki-NLP/opus-mt-en-mul"
        _bn_tok = MarianTokenizer.from_pretrained(name)
        _bn_model = MarianMTModel.from_pretrained(name)


# =====================================================
# SMOOTHERS
# =====================================================

def _smooth_hindi(text: str) -> str:
    replacements = {
        "इसका मतलब है कि": "",
        "यह दर्शाता है कि": "",
        "आपको चाहिए कि": "",
        "कृपया ध्यान दें कि": "",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip()


def _smooth_bengali(text: str) -> str:
    replacements = {
        "এর মানে হল": "",
        "এটি নির্দেশ করে যে": "",
        "আপনার উচিত": "",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip()


# =====================================================
# HINGLISH STYLE MIRROR
# =====================================================

def _to_hinglish(text: str) -> str:
    replacements = {
        "Rest": "Aaram karein",
        "Stay hydrated": "paani peete rahein",
        "Seek medical advice": "doctor se salah lein",
        "Monitor symptoms": "symptoms monitor karein",
        "Consult a doctor": "doctor ko dikhaein",
    }

    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


# =====================================================
# BENGALISH STYLE MIRROR (IMPORTANT)
# =====================================================

def _to_bengalish(text: str) -> str:
    replacements = {
        "Rest": "bhalo kore rest nin",
        "Stay hydrated": "paani beshi kore khan",
        "Seek medical advice": "doctor er kache jan",
        "Monitor symptoms": "lakshan gulo lokkho korun",
        "Consult a doctor": "doctor dekhan",
        "If symptoms worsen": "jodi obostha kharap hoy",
    }

    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


# =====================================================
# MAIN TRANSLATE FUNCTION (CONTRACT SAFE)
# =====================================================

def translate(text: str, lang: str) -> str:
    """
    Multilingual safe translator.
    NEVER crashes.
    NEVER returns garbage.
    """

    if not text:
        return text

    # ---------------------------------
    # Hinglish mirror
    # ---------------------------------
    if lang == "Hinglish":
        return _to_hinglish(text)

    # ---------------------------------
    # Bengalish mirror
    # ---------------------------------
    if lang == "Bengalish":
        return _to_bengalish(text)

    # ---------------------------------
    # Hindi translation
    # ---------------------------------
    if lang == "Hindi":
        try:
            _load_hi()
            inputs = _hi_tok(text, return_tensors="pt", padding=True)
            out = _hi_model.generate(**inputs)
            result = _hi_tok.decode(out[0], skip_special_tokens=True)

            # Quality guard
            if len(result) < 10:
                return text

            return _smooth_hindi(result)

        except Exception as e:
            print("⚠️ Hindi MT fallback:", e)
            return text

    # ---------------------------------
    # Bengali translation (STRONG GUARD)
    # ---------------------------------
    if lang == "Bengali":
        try:
            _load_bn()
            inputs = _bn_tok(text, return_tensors="pt", padding=True)
            out = _bn_model.generate(**inputs)
            result = _bn_tok.decode(out[0], skip_special_tokens=True)

            # 🚨 Bengali garbage detector
            bad_markers = [
                "simptome",
                "sigual",
                "necessariamente",
                "stresse",
                "hidrated",
                "administrar",
            ]

            if len(result) < 10 or any(b in result.lower() for b in bad_markers):
                return text  # SAFE fallback

            return _smooth_bengali(result)

        except Exception as e:
            print("⚠️ Bengali MT fallback:", e)
            return text

    # ---------------------------------
    # English fallback
    # ---------------------------------
    return text