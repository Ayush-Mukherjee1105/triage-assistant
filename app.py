from flask import Flask, render_template, request
from utils.language import normalize_text, detect_language
from utils.translator import translate
from utils.duration_rules import extract_duration_days, duration_bucket
from utils.severity import assess_severity
from utils.diagnosis_sanity import correct_diagnosis
from utils.xai_explainer import explain, visualize_explanation
from model.red_flag_predict import predict_red_flag
from model.diagnosis_predict import predict_diagnosis
from model.label_map import decode_label
from knowledge_graph.kg import enrich_diagnosis
# Flask init
app = Flask(__name__)
# HOME
@app.route("/")
def home():
    return render_template("index.html")
# ANALYZE
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        raw_text = request.form.get("text", "").strip()
        if not raw_text:
            return render_template(
                "index.html",
                error="Please enter symptoms."
            )
        # LANGUAGE DETECTION
        detected_lang = detect_language(raw_text)
        # NORMALIZATION
        normalized = normalize_text(raw_text)
        # RED FLAG
        is_red, rf_label, rf_conf = predict_red_flag(normalized)
        # DURATION
        days = extract_duration_days(normalized)
        duration_text = duration_bucket(days)
        # DIAGNOSIS
        enhanced_text = f"{normalized} duration {duration_text}"
        diag_label, diag_conf = predict_diagnosis(enhanced_text)
        # clinical sanity correction
        diag_label = correct_diagnosis(diag_label, normalized)
        # SEVERITY
        severity, severity_score = assess_severity(
            is_red_flag=is_red,
            diagnosis_confidence=diag_conf,
            duration_days=days,
            text=normalized
        )
        # LABEL + KNOWLEDGE GRAPH
        diag_name = decode_label(diag_label)
        kg_info = enrich_diagnosis(diag_label)
        desc = kg_info.get("description", diag_name)
        if is_red:
            action = "⚠️ URGENT: Seek immediate medical care or go to the nearest emergency facility."
        else:
            action = kg_info.get(
                "action",
                "Consult a healthcare professional if symptoms persist."
            )
        # MULTILINGUAL MIRRORING
        desc = translate(desc, detected_lang)
        action = translate(action, detected_lang)

        # -----------------------------
        # XAI
        # -----------------------------

        try:

            xai_tokens = explain(normalized)
            xai_image = visualize_explanation(normalized)

        except Exception as e:

            print("XAI ERROR:", e)
            xai_tokens = []
            xai_image = None

        # -----------------------------
        # RESULT OBJECT
        # -----------------------------

        result = {
            "severity": severity,
            "severity_score": severity_score,
            "red_flag": rf_label,
            "duration": duration_text,
            "diagnosis": diag_name,
            "confidence": round(diag_conf, 3),
            "description": desc,
            "action": action,
            "language": detected_lang,
            "xai_tokens": xai_tokens,
            "xai_image": xai_image
        }

        return render_template("index.html", result=result)

    except Exception as e:

        print("ANALYZE ERROR:", e)

        return render_template(
            "index.html",
            error="System error occurred. Please try again."
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)