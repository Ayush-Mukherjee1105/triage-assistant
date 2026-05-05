import json
from datetime import datetime
from pathlib import Path

AUDIT_PATH = Path("audit_logs")

AUDIT_PATH.mkdir(exist_ok=True)

LOG_FILE = AUDIT_PATH / "triage_log.jsonl"


def log_decision(original_text, response_payload):
    """
    Append triage decision to audit log.

    JSONL format:
    one record per line
    """

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "original_input": original_text,
        "result": response_payload
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Audit log failed:", e)
