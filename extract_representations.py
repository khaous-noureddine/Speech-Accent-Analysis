from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import transformers
import pickle
import lzma

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from loguru import logger
from tqdm.auto import tqdm

tqdm.pandas()
transformers.logging.set_verbosity_error()


class AudioRepresentationExtractor(ABC):
    @abstractmethod
    def load(self, audio_filepath: Path) -> tuple[np.ndarray, int]:
        """
        Load a file from a `Path` and return the audio data and the sampling rate.
        """
        pass

    @abstractmethod
    def extract_all_layers(
        self, speech_signal: np.ndarray, sampling_rate: int
    ) -> dict:
        pass

    def get_representation(
        self,
        audio_filepath: Path,
        layer="all",
        pooling: str = None,
        return_attentions: bool = True,
    ) -> np.ndarray:
        assert pooling is None, "Pooling is not implemented yet!"

        audio, sr = self.load(audio_filepath)
        all_layers = self.extract_all_layers(audio, sr)

        if layer != "all" or pooling is not None:
            raise Exception("Not implemented!")

        return (all_layers["attentions"] if return_attentions else {}) | all_layers[
            "embbedings"
        ]

    def select_device(self):
        if torch.cuda.is_available():
            logger.info("using cuda")
            return "cuda"
        elif torch.backends.mps.is_built():
            logger.info("using mps")
            return "mps"
        else:
            logger.info("falling back to cpu")
            return "cpu"

    def load_from_dataframe(self, df, filename: str, return_attentions: bool = True):
        return pd.concat(
            [
                df,
                df.progress_apply(
                    lambda x: self.get_representation(
                        x[filename], return_attentions=return_attentions
                    ),
                    result_type="expand",
                    axis=1,
                ),
            ],
            axis=1,
        )
       
    def load_from_dataframe_and_store(self, df, filename, output_dir, compress=False):
        def predict(row):
            file_path = row[filename]
            # Extraction de l'identifiant (ex: 'arabic12' de 'arabic12_chunk01')
            file_name = file_path.stem.split('_')[0] 

            emb_filename = (
                output_dir
                / f"{file_path.stem}_embeddings.pkl{'.xz' if compress else ''}"
            )

            att_filename = (
                output_dir
                / f"{file_path.stem}_attentions.pkl{'.xz' if compress else ''}"
            )

            # Extraction si les fichiers n'existent pas encore
            if not emb_filename.is_file() or not att_filename.is_file():
                res = self.get_representation(row[filename], return_attentions=True)

            if not emb_filename.is_file():
                # On filtre pour ne garder que les couches d'embeddings
                emb = {
                    key: item
                    for key, item in res.items()
                    if key.startswith("embedding")
                }
                
                # AJOUT : On insère l'identifiant pour le merge futur
                emb["filename"] = file_name
                # On peut aussi garder le chemin complet si besoin
                emb["filepath"] = str(file_path)

                # Sauvegarde avec ou sans compression
                if not compress:
                    with open(emb_filename, "wb") as f:
                        pickle.dump(emb, f)
                else:
                    with lzma.open(emb_filename, "wb") as f:
                        pickle.dump(emb, f)

        df.progress_apply(predict, axis=1)
    
class XLSR53RepresentationExtractor(AudioRepresentationExtractor):
    def __init__(self, **kwargs):
        """
        Load the XLSR-53 Large model.

        Parameters
        ----------
        - device: where the model must be loaded (can either be "cpu", "cuda", "mps" or "auto")
        """
        device = kwargs.get("device", "auto")
        model = kwargs.get("model", "facebook/wav2vec2-large-xlsr-53")

        if device == "auto":
            device = self.select_device()
            logger.error(f"auto-detect device: using {device}!")

        self.device = device
        self.model = Wav2Vec2Model.from_pretrained(model).to(self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)

    def load(self, audio_filepath: Path) -> np.ndarray:
        """
        Load an audio file and ensure that it can be used in XLS-R/wav2vec2

        To be used by XLS-R, wav files have to:
        - have a sampling rate of 16_000Hz
        - use only one channel (mono file)
        ⤷ by default, librosa.load convert file to mono & resample data

        Parameters:
        - audio_filepath, str → path to the file
        """
        return librosa.load(audio_filepath, sr=16_000)

    def extract_all_layers(self, speech_signal: np.ndarray, sampling_rate: int) -> dict:
        """
        Compute the representations

        Parameters
        ----------
        - speech_signal: np.ndarray: the speech signal. Make sure that the signal has the properties expected by the model by using the `load` method.
        """
        # ensure that we are in evaluation mode (in particular drop-out layer are not activated)
        assert not self.model.training
        assert sampling_rate == 16_000

        inputs = self.feature_extractor(
            speech_signal, sampling_rate=sampling_rate, return_tensors="pt"
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(
                **inputs, output_hidden_states=True, output_attentions=True
            )

            # `output.attentions` is a list of 24 tensors (one for each layer). Each tensor has a shape # [batch_size ✕ n_heads ✕ seq_length ✕ seq_length] where n_heads is 16 and batch_size is 1.
            # ⤷ we make the layer and head number information explicit
            attentions = {}# {
            #    f"attention_layer_{layer}_-_head_{i}": att.squeeze()[i]
            #    .cpu()
            #    .detach()
            #    .numpy()
            #    for layer, att in enumerate(output.attentions)
            #    for i in range(att.shape[1])
            #}

            # `output.hidden_states` is a list of 25 tensors (input embeddings + 24 encoder layers)
            # Each tensor has a shape of [batch_size ✕ sequence_length ✕ repr_size]
            #
            # batch size is always 1 — we are considering a single audio segment.
            # repr_size is 1,024 for XLSR-53
            # the encoder outputs representation at 49Hz
            # ⤷ sequence length is equal to 49 ✕ number of seconds in signal

            # detach the tensor and remove the batch_dimension
            return {
                "embbedings": {
                    f"embedding_layer_{layer}": emb.squeeze().cpu().detach().numpy()
                    for layer, emb in enumerate(output.hidden_states)
                },
                "attentions": attentions,
            }


def filter_files_by_max_size(files: list[Path], max_size_bytes: int = 100 * 1024 * 1024) -> list[Path]:
    """
    Returns a list of files smaller than the given max size (in bytes).

    :param files: List of Path objects representing files.
    :param max_size_bytes: Maximum allowed file size in bytes (default is 100 MB).
    :return: List of Path objects whose size is less than max_size_bytes.
    """
    if max_size_bytes is None:
        return files
    return [f for f in files if f.stat().st_size < max_size_bytes]





if __name__ == "__main__":
    import argparse

    from itertools import chain
    from functools import partial

    all_models = {
        "xlsr53": partial(XLSR53RepresentationExtractor, model="facebook/wav2vec2-large-xlsr-53"),
        "wav2vec2-english": partial(XLSR53RepresentationExtractor, model="facebook/wav2vec2-base-960h"),
        # "mfcc": MFCCRepresentationExtractor,
        # "whisper": WhisperRepresentationExtractor,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=all_models.keys())
    parser.add_argument("--max_n_files", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--exclude_subdirectories", action="store_true", default=False)
    parser.add_argument("--max_size", type=int, default=100*1024*1024)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    glob = (
        args.corpus_dir.glob if args.exclude_subdirectories else args.corpus_dir.rglob
    )

    df = pd.DataFrame({"filename": filter_files_by_max_size(list(chain(glob("*.wav"), glob("*.mp3"))), args.max_size)})

    if args.max_n_files is not None:
        df = df.head(args.max_n_files)
        # df = df.sample(n=args.max_n_files)
    print(df)
    logger.info("Loading model")
    extractor = all_models[args.model](device=args.device)

    logger.info("Extracting representations")
    extractor.load_from_dataframe_and_store(df, "filename", args.output_dir)