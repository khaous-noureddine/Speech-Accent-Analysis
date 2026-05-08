"""
utils.py — shared text normalization for Arctic & L2-Arctic imports.
"""
 
import re
import soundfile as sf
import numpy as np
import torch
import torchaudio
from pathlib import Path
 
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
    text = re.sub(r"\s+", " ", text)       # collapse multiple spaces into one
    return text

def load_audio(path: str, target_sr: int = 16000, max_len_samples: int = None) -> torch.Tensor:
    waveform, sr = sf.read(path, dtype="float32")  # [T] ou [T, C]

    # Stereo → mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = torch.from_numpy(waveform)  # [T]

    # Resample if necessary
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Truncate
    if max_len_samples is not None and waveform.shape[0] > max_len_samples:
        waveform = waveform[:max_len_samples]

    return waveform


def load_with_librosa(path: Path, target_sr: int = 16000, max_len_samples: int = None) -> torch.Tensor:
    import librosa
    waveform, sr = librosa.load(path, sr=target_sr, mono=True)  # [T]

    waveform = torch.from_numpy(waveform)  # [T]

    # Truncate
    if max_len_samples is not None and waveform.shape[0] > max_len_samples:
        waveform = waveform[:max_len_samples]

    return waveform