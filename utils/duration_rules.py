import re

# -----------------------------
# word numbers
# -----------------------------

NUMBER_WORDS = {
    "one":1,
    "two":2,
    "three":3,
    "four":4,
    "five":5,
    "six":6,
    "seven":7,
    "eight":8,
    "nine":9,
    "ten":10
}

# -----------------------------
# phrase rules (highest priority)
# -----------------------------

PHRASE_RULES = {

    # English
    "today":1,
    "since yesterday":2,
    "yesterday":2,

    "few days":3,
    "several days":5,
    "couple days":2,

    "over a week":8,
    "more than a week":8,
    "about a week":7,
    "around a week":7,
    "last week":7,
    "past week":7,

    "several weeks":21,
    "few weeks":14,

    "several months":120,
    "few months":90,

    # Hinglish
    "kal se":2,
    "aaj se":1,
    "pichle hafte":7,
    "ek hafte se":7,
    "kai din":4,
    "kaafi din":5,

    # Hindi
    "एक दिन":1,
    "दो दिन":2,
    "तीन दिन":3,
    "कुछ दिन":4,
    "कई दिन":5,
    "एक हफ्ते":7,
    "एक सप्ताह":7,
    "कई हफ्ते":21,
    "कई महीने":120,

    # Bengalish
    "kal theke":2,
    "aj theke":1,
    "1 week dhore":7,
    "ek week dhore":7,
    "onek din":5,
    "koyek din":4,
    "koyek week":21,

    # Bengali
    "একদিন":1,
    "দুইদিন":2,
    "তিনদিন":3,
    "কয়েকদিন":4,
    "অনেকদিন":5,
    "এক সপ্তাহ":7,
    "গত সপ্তাহ":7,
    "কয়েক সপ্তাহ":21
}

# -----------------------------
# main extraction
# -----------------------------

def extract_duration_days(text):

    text = text.lower()

    # -------------------------
    # phrase matches
    # -------------------------

    for phrase, days in PHRASE_RULES.items():

        if phrase in text:
            return days

    # -------------------------
    # numeric days
    # -------------------------

    m = re.search(r'(\d+)\s*(day|days)', text)
    if m:
        return int(m.group(1))

    # -------------------------
    # numeric weeks
    # -------------------------

    m = re.search(r'(\d+)\s*(week|weeks)', text)
    if m:
        return int(m.group(1)) * 7

    # -------------------------
    # numeric months
    # -------------------------

    m = re.search(r'(\d+)\s*(month|months)', text)
    if m:
        return int(m.group(1)) * 30

    # -------------------------
    # numeric years
    # -------------------------

    m = re.search(r'(\d+)\s*(year|years)', text)
    if m:
        return int(m.group(1)) * 365

    # -------------------------
    # word numbers
    # -------------------------

    for word, val in NUMBER_WORDS.items():

        if f"{word} day" in text:
            return val

        if f"{word} week" in text:
            return val * 7

        if f"{word} month" in text:
            return val * 30

        if f"{word} year" in text:
            return val * 365

    # -------------------------
    # generic keywords
    # -------------------------

    if "week" in text:
        return 7

    if "month" in text:
        return 30

    if "year" in text:
        return 365

    if "days" in text:
        return 3

    # -------------------------
    # fallback
    # -------------------------

    return 1

def duration_bucket(days):
    if days <= 3:
        return "acute"
    elif days <= 7:
        return "short_term"
    elif days <= 14:
        return "persistent"
    else:
        return "chronic"