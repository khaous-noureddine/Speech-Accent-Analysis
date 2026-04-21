"""
text_utils.py — shared text normalization for Arctic & L2-Arctic imports.
"""
 
import re
 
 
def normalize_transcript(text: str) -> str:
    """
    Normalize a transcript so that ARCTIC and L2-ARCTIC match exactly.
 
    Steps:
        1. Strip leading/trailing whitespace
        2. Lowercase
        3. Remove punctuation except apostrophes  (don't → don't, not don t)
        4. Collapse multiple spaces into one
    """
    text = text.strip()
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)   # keep word chars, spaces, apostrophes
    text = re.sub(r"\s+", " ", text)
    return text