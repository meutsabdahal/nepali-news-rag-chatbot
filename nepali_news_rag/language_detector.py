from langdetect import detect


def _devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    devanagari_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    return devanagari_chars / len(text)


def detect_language(text: str) -> str:
    if _devanagari_ratio(text) >= 0.2:
        return "Nepali"

    try:
        return "English" if detect(text) == "en" else "Nepali"
    except Exception:
        return "Nepali"
