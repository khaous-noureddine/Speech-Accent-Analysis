"""
utils.py — shared text normalization for Arctic & L2-Arctic imports.
"""
 
import re
import torch
import torchaudio
 
 
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


def load_audio(path: str, target_sr: int = 16000, max_len_samples: int = None) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.squeeze(0)
    if max_len_samples is not None and waveform.shape[0] > max_len_samples:
        waveform = waveform[:max_len_samples]
    return waveform