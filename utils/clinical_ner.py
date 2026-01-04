import re

def extract_clinical_entities(text: str):
    text = text.lower()

    entities = {
        "loss_of_consciousness": False,
        "memory_loss": False,
        "vomiting": False,
        "severe_pain": False,
        "fever_days": 0
    }

    if re.search(r"passed out|unconscious|fainted", text):
        entities["loss_of_consciousness"] = True

    if re.search(r"don'?t remember|memory loss|confused", text):
        entities["memory_loss"] = True

    if "vomit" in text:
        entities["vomiting"] = True

    if "severe pain" in text or "worst pain" in text:
        entities["severe_pain"] = True

    fever_match = re.search(r"fever.*?(\d+)\s*day", text)
    if fever_match:
        entities["fever_days"] = int(fever_match.group(1))

    return entities
