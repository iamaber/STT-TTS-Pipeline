def clean_text_for_display(text: str) -> str:
    cleaned = "".join(char for char in text if 32 <= ord(char) <= 126 or char in "\n\t")

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()
