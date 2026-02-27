"""
Unified Extraction + Alignment Pipeline
========================================

Au lieu de faire :
1. Extraire embeddings → DataFrame géant
2. Sauvegarder
3. Recharger
4. Aligner
5. Re-sauvegarder par fichier

On fait TOUT d'un coup :
1. Extraire embeddings
2. Aligner directement avec TextGrid
3. Sauvegarder le résultat aligné

Résultat : Un seul fichier par (audio, layer) déjà aligné et prêt à l'emploi !
"""

from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import librosa
import transformers
import pickle
import textgrid

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from loguru import logger
from tqdm.auto import tqdm

tqdm.pandas()
transformers.logging.set_verbosity_error()


# ============================================================================
# HELPERS
# ============================================================================

def read_textgrid(tg_path):
    """Load TextGrid file with words and phones annotations."""
    if not tg_path.exists():
        return {"words": [], "phones": []}
    try:
        t = textgrid.TextGrid.fromFile(str(tg_path))
        return {name: tiers for name, tiers in zip(t.getNames(), t.tiers)}
    except Exception:
        return {"words": [], "phones": []}


def align_single_layer(embeddings, annotations, freq, label_column="words"):
    """
    Aligne UNE layer d'embeddings avec les annotations.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Matrice (n_frames, embedding_dim)
    annotations : list of Interval
        Liste d'objets Interval du TextGrid
    freq : float
        Fréquence temporelle (ex: 0.02 pour Whisper = 50Hz)
    label_column : str
        'words' ou 'phones'
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes: repr, time, start_time, end_time, annotation
    """
    # Créer le DataFrame des annotations
    df_annotations = pd.DataFrame([
        {
            "start_time": interval.minTime,
            "end_time": interval.maxTime,
            "annotation": interval.mark,
        }
        for interval in annotations
    ])
    
    # Créer le DataFrame des embeddings avec timestamps
    n_frames = embeddings.shape[0]
    df_embeddings = pd.DataFrame({
        "repr": list(embeddings),  # Chaque ligne = un vecteur
        "time": np.arange(n_frames) * freq
    })
    
    # Aligner !
    df_aligned = pd.merge_asof(
        df_embeddings,
        df_annotations,
        left_on="time",
        right_on="start_time"
    )
    
    return df_aligned


# ============================================================================
# EXTRACTEUR WHISPER
# ============================================================================

class WhisperExtractorAligner:
    """Extrait les embeddings Whisper ET les aligne directement."""
    
    def __init__(self, device="auto", gpu_id=0):
        model_name = "openai/whisper-large-v3"
        
        if device == "auto":
            self.device = self._select_device(gpu_id)
        else:
            self.device = device
            
        logger.info(f"Loading Whisper model on {self.device}")
        
        self.feature_extractor = AutoProcessor.from_pretrained(model_name)
        full_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(self.device)
        self.model = full_model.get_encoder()
        
        # Whisper : 50 Hz (un frame toutes les 0.02 secondes)
        self.freq = 0.02
        
    def _select_device(self, gpu_id=0):
        if torch.cuda.is_available():
            return f"cuda:{gpu_id}"
        elif torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"
    
    def extract_all_layers(self, audio_path: Path):
        """Extrait les 33 layers d'embeddings."""
        # Charger l'audio
        speech_signal, sr = librosa.load(audio_path, sr=16_000)
        
        # Extraire les features
        features = self.feature_extractor(
            speech_signal,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        input_features = features.input_features.to(self.device, dtype=torch_dtype)
        
        # Forward pass
        with torch.no_grad():
            encoder_hidden_states = self.model(
                input_features,
                output_hidden_states=True
            ).hidden_states
        
        # Retourner les 33 layers (tuple → dict)
        return {
            f"layer_{i}": state.detach().float().cpu().numpy().squeeze()
            for i, state in enumerate(encoder_hidden_states)
        }
    
    def process_file(
        self,
        audio_path: Path,
        textgrid_path: Path,
        metadata: dict,
        annotation_type: str = "words"
    ):
        """
        Traite UN fichier audio : extraction + alignement.
        
        Parameters
        ----------
        audio_path : Path
            Chemin vers le fichier audio
        textgrid_path : Path
            Chemin vers le TextGrid
        metadata : dict
            Métadonnées du speaker (age, country, native_language, etc.)
        annotation_type : str
            'words' ou 'phones'
            
        Returns
        -------
        dict
            {
                'layer_0': DataFrame aligné,
                'layer_1': DataFrame aligné,
                ...
            }
        """
        # 1. Extraire les embeddings
        logger.info(f"Extracting embeddings for {audio_path.name}")
        all_layers = self.extract_all_layers(audio_path)
        
        # 2. Charger le TextGrid
        tg_data = read_textgrid(textgrid_path)
        annotations = tg_data.get(annotation_type, [])
        
        if not annotations:
            logger.warning(f"No {annotation_type} found in {textgrid_path}")
            return {}
        
        # 3. Aligner chaque layer
        aligned_layers = {}
        
        for layer_name, embeddings in tqdm(all_layers.items(), desc="Aligning layers", leave=False):
            df_aligned = align_single_layer(
                embeddings,
                annotations,
                self.freq,
                label_column=annotation_type
            )
            
            # Ajouter les métadonnées
            df_aligned["layer"] = layer_name
            df_aligned["filename"] = audio_path.stem
            df_aligned["speaker"] = metadata.get("speaker", "")
            df_aligned["age"] = metadata.get("age", np.nan)
            df_aligned["country"] = metadata.get("country", "")
            df_aligned["native_language"] = metadata.get("native_language", "")
            df_aligned["sex"] = metadata.get("sex", "")
            
            aligned_layers[layer_name] = df_aligned
        
        return aligned_layers


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_unified_pipeline(
    corpus_dir: Path,
    textgrid_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    annotation_type: str = "words",
    max_n_files: int = None,
    device: str = "auto",
    gpu_id: int = 0
):
    """
    Pipeline unifié : extraction + alignement en un seul passage.
    
    Structure de sortie :
    output_dir/
        layer_0/
            arabic29_aligned.pkl
            english150_aligned.pkl
            ...
        layer_1/
            ...
    """
    # 1. Lister les fichiers audio
    audio_files = list(corpus_dir.rglob("*.wav")) + list(corpus_dir.rglob("*.mp3"))
    
    if max_n_files:
        audio_files = audio_files[:max_n_files]
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # 2. Charger les métadonnées
    metadata_df = pd.read_csv(metadata_path, sep="\t")
    metadata_df["speaker"] = metadata_df["filename"].str.upper()
    metadata_dict = metadata_df.set_index("speaker").to_dict("index")
    
    # 3. Initialiser l'extracteur
    extractor = WhisperExtractorAligner(device=device, gpu_id=gpu_id)
    
    # 4. Traiter chaque fichier
    for audio_path in tqdm(audio_files, desc="Processing files"):
        # Trouver le TextGrid correspondant
        textgrid_path = textgrid_dir / f"{audio_path.stem}.TextGrid"
        
        # Récupérer les métadonnées
        speaker_id = audio_path.stem.split("_")[0].upper()
        metadata = metadata_dict.get(speaker_id, {})
        
        # Extraire + aligner
        try:
            aligned_layers = extractor.process_file(
                audio_path,
                textgrid_path,
                metadata,
                annotation_type=annotation_type
            )
            
            # Sauvegarder par layer
            for layer_name, df_aligned in aligned_layers.items():
                layer_dir = output_dir / layer_name
                layer_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = layer_dir / f"{audio_path.stem}_aligned.pkl"
                df_aligned.to_pickle(output_file)
                
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            continue
    
    logger.info("✅ Pipeline completed!")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified extraction + alignment pipeline"
    )
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--textgrid_dir", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--annotation", choices=["words", "phones"], default="words")
    parser.add_argument("--max_n_files", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    
    run_unified_pipeline(
        corpus_dir=args.corpus_dir,
        textgrid_dir=args.textgrid_dir,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        annotation_type=args.annotation,
        max_n_files=args.max_n_files,
        device=args.device,
        gpu_id=args.gpu_id
    )