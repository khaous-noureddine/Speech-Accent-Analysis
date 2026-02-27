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
import textgrid


from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from loguru import logger
from tqdm.auto import tqdm

tqdm.pandas()

transformers.logging.set_verbosity_error()


def read_textgrid(tg_path):
    if not tg_path.exists():
        return {"words": [], "phones": []}
    try:
        t = textgrid.TextGrid.fromFile(str(tg_path))
        return {name: tiers for name, tiers in zip(t.getNames(), t.tiers)}
    except Exception:
        return {"words": [], "phones": []}
    

def align_single_layer(embeddings, annotations, freq, metadata):
    """
    Aligne UNE layer d'embeddings avec les annotations.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Matrice (n_frames, embedding_dim)
    annotations : list of Interval
        Liste d'objets Interval du TextGrid
    freq : float
        Fréquence temporelle (0.02 pour Whisper = 50Hz)
    metadata : dict
        Métadonnées du fichier (speaker, age, country, etc.)
    
    Returns
    -------
    pd.DataFrame
        DataFrame aligné prêt à sauvegarder
    """
    # 1. Créer DataFrame des annotations
    df_annotations = pd.DataFrame([
        {
            "start_time": interval.minTime,
            "end_time": interval.maxTime,
            "annotation": interval.mark,
        }
        for interval in annotations
        if interval.mark.strip()  # Ignorer les annotations vides
    ])
    
    if df_annotations.empty:
        return pd.DataFrame()  # Retourner DataFrame vide si pas d'annotations
    
    # 2. Créer DataFrame des embeddings avec timestamps
    n_frames = embeddings.shape[0]
    df_embeddings = pd.DataFrame({
        "repr": list(embeddings),  # Chaque ligne = un vecteur
        "time": np.arange(n_frames) * freq
    })
    
    # 3. Aligner avec merge_asof
    df_aligned = pd.merge_asof(
        df_embeddings,
        df_annotations,
        left_on="time",
        right_on="start_time"
    )
    
    # 4. Ajouter les métadonnées
    for key, value in metadata.items():
        df_aligned[key] = value
    
    return df_aligned
    
    
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

        att = all_layers.get("attentions", {}) if return_attentions else {}
        return att | all_layers["embbedings"]
        # return (all_layers["attentions"] if return_attentions else {}) | all_layers[
        #     "embbedings"
        # ]

    def select_device(self, gpu_id: int = 0) -> str:
        if torch.cuda.is_available():
            logger.info(f"using cuda:{gpu_id}")
            return f"cuda:{gpu_id}"
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
        
    def extract_to_dataframe(self, df, filename_col: str, return_attentions: bool = True):
        """
        Extract embeddings directly into a DataFrame without saving individual files.
        """
        def extract_row(row):
            file_path = row[filename_col]
            file_name = file_path.stem.split('_')[0]
            
            try:
                # Extract embeddings
                res = self.get_representation(file_path, return_attentions=return_attentions)
                
                # Transform 'embedding_layer_X' -> 'layer_X'
                transformed = {
                    k.replace("embedding_layer_", "layer_") if k.startswith("embedding_layer_") else k: v 
                    for k, v in res.items()
                }
                
                # Add metadata
                transformed["filename"] = file_name
                transformed["filepath"] = str(file_path)
                
                return pd.Series(transformed)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return pd.Series({"filename": file_name, "filepath": str(file_path)})
        
        return df.progress_apply(extract_row, axis=1)


class WhisperRepresentationExtractor(AudioRepresentationExtractor):
    def __init__(self, **kwargs):
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        # model_name = "distil-whisper/distil-large-v2"
        model_name = "openai/whisper-large-v3"

        self.device = kwargs.get("device", "auto")
        if self.device == "auto":
            self.device = self.select_device(kwargs.get("gpu_id", 0))

        self.feature_extractor = AutoProcessor.from_pretrained(model_name)
        full_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(self.device)

        # whisper models have an encoder-decoder architecture (the model can be used for transcription without any finetuning). Here we are only interested in the encoder part of the model that build the audio representation.
        # see: https://github.com/huggingface/distil-whisper/issues/67
        self.model = full_model.get_encoder()

    def load(self, audio_filepath: Path) -> np.ndarray:
        """
        The Whisper feature extractor expects audio inputs with a sampling rate of 16kHz and must have a length of 30 s max. If the audio is longer than 30 s it will truncated, if it is shorter is will be padded.

        Sources:
        - https://huggingface.co/blog/fine-tune-whisper
        """
        # assert (
        #     librosa.get_duration(path=audio_filepath) < 30
        # ), f"{audio_filepath} longer than 30s! (duration={librosa.get_duration(path=audio_filepath)} s)"

        return librosa.load(audio_filepath, sr=16_000)

    def extract_all_layers(self, speech_signal: np.ndarray, sr: int) -> dict:
        features = self.feature_extractor(
            speech_signal, return_attention_mask=True, return_tensors="pt"
        )

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # attention_mask is of size batch_size ✕ 3,000 (everything is padded to 30s)
        # ⤷ 0s for padding positions, 1s everywhere else
        #
        # for the moment we assume that we are encoding one segment after another without considering batches. That is why we can get ride of the first dimension in the masks.
        attention_mask = features.attention_mask.numpy().squeeze()

        # input_features is of shape batch_size ✕ 128 ✕ 3,000 (with batch_size = 1 for the moment)
        # ⤷ 128 number of features (log-magnitude Mel spectrogram representation of the input, The
        # number of Mel bins is a design choice that is defined in the configuration file of the model
        # (`num_mel_bins parameter`)
        # ⤷ 3,000 is the sequence length and is a fixed number (everything is padded!)
        input_features = features.input_features.to(self.device, dtype=torch_dtype)

        with torch.no_grad():
            # the encoder has 32 layers
            # we therefore gets a tuple of 33 tensor of size 1 ✕ 1,500 ✕ 1,280
            # → we are considering a single file, padded to 30s with representation of dimension 1,280
            encoder_hidden_states = self.model(
                input_features, output_hidden_states=True
            ).hidden_states

        # As above: we assume that batch_size is always 1 and get ride of the corresponding dimension
        #
        # For each layer the representation has size 1 ✕ 1,500 ✕ 1,280 (batch_size ✕ sequence_length ✕
        # embedding size). All audios are encoded on the same number of elements and it is not clear
        # (for now I hope) if there is a mask to apply to select only the relevant elements or if all
        # embeddings contain relevant information.
        return {
            "embbedings": {
                f"embedding_layer_{layer}":
            # state.numpy().squeeze()
            state.detach().float().cpu().numpy().squeeze()
            for layer, state in enumerate(encoder_hidden_states)}
        }


class MFCCRepresentationExtractor(AudioRepresentationExtractor):
    def __init__(self, **kwargs):
        self.n_coeffs = kwargs.get("n_coeffs", 48)
        logger.warning("The number of MFCC coefficient is hard-coded!")

    def load(self, audio_filepath: Path) -> tuple[np.ndarray, int]:
        """
        Load a file from a `Path` and return the audio data and the sampling rate.
        """

        # We resample the audio to 16 kHz to consider representations comparable to those of XLSR-53 and
        # because this corresponds to the sampling frequency generally considered (see the detailed
        # explanation in the following function)
        audio, sr = librosa.load(audio_filepath, sr=16_000)
        return audio, sr

    def extract_all_layers(self, speech_signal: np.ndarray, sr: int) -> np.ndarray:

        # In order to guarantee consistency with the representations constructed by neural networks, we
        # consider that MFCCs are composed of a single “layer”.
        #
        # MFCCs are determined from a sliding window on the audio signal. The window has a fixed size
        # (in number of frames) and is shifted by a number of frames at each step/iteration (a parameter
        # often called hop_length). These two parameters are expressed in number of frames; their
        # correlation with a duration will therefore depend on the sampling frequency.
        #
        # In practice, three sampling frequencies are used depending on the type of signal we want to
        # process:
        # - for speech, a sampling frequency of 16 kHz is a good compromise between the quality of the
        # information extracted and the size of the representations constructed
        # - to analyse musical sounds or rich environments, you need to use a higher frequency, such as
        # 22.05 kHz or 44.1 kHz.
        #
        # These values are determined by very simple reasoning:
        # - For speech: Most of the relevant energy is between 300 Hz and 8000 Hz.
        # - For music: A wider range, up to 20 kHz, may be necessary.
        # Nyquist's theorem , one of the fundamental results of information theory, shows that the
        # sampling frequency must be at least 2 times the maximum frequency to be analyzed.

        # we need to transpose the result of librosa MFCC method to have the “representations” as other
        # methods: first dimension = time, second = features
        return {
            "embbedings": {
                "embedding_layer_1": librosa.feature.mfcc(
                    y=speech_signal, sr=sr, n_mfcc=self.n_coeffs
                ).T
            }
        }


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
            device = self.select_device(kwargs.get("gpu_id", 0))
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


class HubertRepresentationExtractor(AudioRepresentationExtractor):
    def __init__(self, **kwargs):
        """
        Load the HuBERT model.

        Parameters
        ----------
        - device: where the model must be loaded (can either be "cpu", "cuda", "mps" or "auto")
        """
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        
        device = kwargs.get("device", "auto")
        model = kwargs.get("model", "facebook/hubert-large-ls960-ft")
        # Alternatives: 
        # - "facebook/hubert-base-ls960" (12 layers, 768-dim)
        # - "facebook/hubert-large-ls960-ft" (24 layers, 1024-dim)

        if device == "auto":
            device = self.select_device(kwargs.get("gpu_id", 0))
            logger.info(f"auto-detect device: using {device}!")

        self.device = device
        self.model = HubertModel.from_pretrained(model).to(self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)

    def load(self, audio_filepath: Path) -> np.ndarray:
        """
        Load an audio file for HuBERT.
        
        HuBERT uses same preprocessing as wav2vec2:
        - sampling rate of 16_000Hz
        - mono file
        """
        return librosa.load(audio_filepath, sr=16_000)

    def extract_all_layers(self, speech_signal: np.ndarray, sampling_rate: int) -> dict:
        """
        Extract all hidden states from HuBERT.
        """
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

            # HuBERT-base: 13 layers (input + 12 encoder layers)
            # HuBERT-large: 25 layers (input + 24 encoder layers)
            # Each tensor: [batch_size × sequence_length × repr_size]
            # repr_size: 768 (base) or 1024 (large)
            # Outputs at 49Hz (same as wav2vec2)
            
            return {
                "embbedings": {
                    f"embedding_layer_{layer}": emb.squeeze().cpu().detach().numpy()
                    for layer, emb in enumerate(output.hidden_states)
                },
                "attentions": {},
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
        "w2v-base": partial(XLSR53RepresentationExtractor, model="facebook/wav2vec2-base-960h"),
        "w2v-large": partial(XLSR53RepresentationExtractor, model="facebook/wav2vec2-large-960h"),
        "mfcc": MFCCRepresentationExtractor,
        "whisper": WhisperRepresentationExtractor,
        "hubert": partial(HubertRepresentationExtractor, model="facebook/hubert-large-ls960-ft"),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=all_models.keys())
    parser.add_argument("--max_n_files", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--exclude_subdirectories", action="store_true", default=False)
    parser.add_argument("--max_size", type=int, default=100*1024*1024)
    parser.add_argument("--metadata_path", required=True, type=Path)
    parser.add_argument("--textgrid_dir", required=True, type=Path)

    args = parser.parse_args()

    # args.output.mkdir(exist_ok=True, parents=True)

    glob = (
        args.corpus_dir.glob if args.exclude_subdirectories else args.corpus_dir.rglob
    )

    df = pd.DataFrame({"filename": filter_files_by_max_size(list(chain(glob("*.wav"), glob("*.mp3"))), args.max_size)})

    if args.max_n_files is not None:
        df = df.head(args.max_n_files)#sample(n=args.max_n_files)
    print(df)

    logger.info("Loading model")
    extractor = all_models[args.model](device=args.device, gpu_id=args.gpu_id)

    # Fréquence d'échantillonnage des embeddings par modèle
    model_freqs = {
        "xlsr53": 1/49,
        "w2v-base": 1/49,
        "w2v-large": 1/49,
        "mfcc": 1/49,
        "whisper": 0.02,  # 50 Hz
        "hubert-base": 1/49,
        "hubert-large": 1/49,
    }
    freq = model_freqs[args.model]
    
    # Charger les métadonnées
    logger.info("Loading metadata")
    metadata = pd.read_csv(args.metadata_path, sep="\t")
    metadata["speaker"] = metadata["filename"].str.upper()
    metadata_dict = metadata.set_index("speaker").to_dict("index")
    
    # Créer le dossier de sortie
    args.output.mkdir(exist_ok=True, parents=True)
    
    # Traiter chaque fichier
    logger.info(f"Processing {len(df)} files")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        audio_path = row["filename"]
        file_stem = audio_path.stem
        speaker_id = file_stem.split("_")[0].upper()
        
        # 1. Extraire les embeddings
        try:
            logger.debug(f"Extracting {file_stem}")
            res = extractor.get_representation(audio_path, return_attentions=False)
        except Exception as e:
            logger.error(f"Error extracting {file_stem}: {e}")
            continue
        
        # 2. Charger le TextGrid
        textgrid_path = args.textgrid_dir / f"{file_stem}.TextGrid"
        tg_data = read_textgrid(textgrid_path)
        
        # Utiliser 'words' par défaut (tu peux ajouter --annotation argument)
        annotations = tg_data.get("words", [])
        
        if not annotations:
            logger.warning(f"No annotations found for {file_stem}")
            continue
        
        # 3. Préparer les métadonnées
        meta = metadata_dict.get(speaker_id, {})
        meta["filename"] = file_stem
        meta["speaker"] = speaker_id
        
        # 4. Aligner chaque layer
        for layer_name, embeddings in res.items():
            if not layer_name.startswith("embedding_layer_"):
                continue
            
            # Renommer embedding_layer_X -> layer_X
            layer_id = layer_name.replace("embedding_layer_", "layer_")
            
            # Aligner
            df_aligned = align_single_layer(embeddings, annotations, freq, meta)
            
            if df_aligned.empty:
                continue
            
            # Ajouter le nom de la layer
            df_aligned["layer"] = layer_id
            
            # Renommer la colonne repr pour cohérence
            # df_aligned = df_aligned.rename(columns={"repr": layer_id})
            
            # 5. Sauvegarder
            layer_dir = args.output / layer_id
            layer_dir.mkdir(exist_ok=True)
            
            output_file = layer_dir / f"{file_stem}_aligned.pkl"
            df_aligned.to_pickle(output_file)
    
    logger.info("✅ Done!")