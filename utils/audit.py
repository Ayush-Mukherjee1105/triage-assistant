# utils/audit.py

import json
from datetime import datetime

def log_decision(input_text, audit):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_text,
        "audit": audit
    }

    with open("audit_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
