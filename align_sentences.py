"""
Algin embeddings with the annotation,
For that we will simply compute at which time each embedding occurs and find the closest time in the annotation.

Output, each wav file is associated to a file containing a dataframe with the embedding aligned with the annotation
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from loguru import logger


def align_representations(row, label_column, embedding_column, freq):
    row = row.to_frame().T.reset_index().copy()

    # Create a DataFrame with the annotations and their timestamp
    # -----------------------------------------------------------
    #
    # for the moment, there is a column (either words or phones) that contain a list of annotations, we will transform the dataframe so that there is one annotation for each row (the annotation is an Interval object describing element in a TextGrid file.)
    # We create a new DataFrame in which there is one element in each row and the different “parts” of the Interval are stored in columns.
    
    # Gestion de l'erreur : on filtre les NaN après l'explode (listes vides)
    exploded_annots = row[[label_column]].explode(column=label_column)
    exploded_annots = exploded_annots[exploded_annots[label_column].notna()]
    
    if exploded_annots.empty:
        return pd.DataFrame()

    annotations = (
        exploded_annots
        .apply(
            lambda x: {
                "start_time": x[label_column].minTime,
                "end_time": x[label_column].maxTime,
                "annotation": x[label_column].mark,
            },
            axis=1,
            result_type="expand",
        )
    )

    # Create a dataframe with the embeddings
    # --------------------------------------
    #
    # For the moment, the embedding is a matrix n_frames × embedding_size. We will create a DataFrame in which each frame is stored in its own row and aligned with the corresponding starting time in the utterance.
    representations = row[[embedding_column]].explode(column=embedding_column)
    representations["time"] = freq
    representations["time"] = representations["time"].cumsum() - freq

    # save metadata
    representations["speaker"] = row["speaker"]
    representations["filename"] = row["filepath"]
    representations["sentence"] = row["sentence"]

    # using the merge_asof method (merge of two DataFrames based on the nearest key) to align the annotation and the embeddings.
    return pd.merge_asof(
        representations, annotations, left_on="time", right_on="start_time"
    )


if __name__ == "__main__":
    # Maps model name to the frequency at which embeddings are built
    # This information is required to align the embeddings to the input signal.
    #
    # XXX is this information stored in HuggingFace 🤗 models
    models = {"xlsr53": 1 / 49}

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=argparse.FileType("rb"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--annotation", default="words")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--model_name", required=True, choices=models.keys())

    args = parser.parse_args()

    df = pd.read_pickle(args.input)

    if args.layer is None:
        layers = df.columns[df.columns.str.startswith("layer_")]
    else:
        layers = [f"layer_{args.layer}"]

    for layer in layers:
        logger.info(f"compute alignments for layer: {layer}")

        output = args.output_dir / layer
        output.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            d = align_representations(
                row,
                label_column=args.annotation,
                embedding_column=layer,
                freq=models[args.model_name],
            )
            
            # Gestion de l'erreur : on passe au suivant si aucune donnée n'est extraite
            if d.empty:
                continue

            output_filename = output / f"{Path(d.iloc[0]['filename']).stem}_aligned.pkl"

            d["layer"] = layer
            d = d.rename(columns={layer: "repr"})
            d.to_pickle(output_filename)