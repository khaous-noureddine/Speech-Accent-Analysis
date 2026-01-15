"""
This script aims to ...
"""


import pandas as pd
import textgrid
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

def read_textgrid(tg_path):
    if not tg_path.exists():
        return {"words": [], "phones": []}
    try:
        t = textgrid.TextGrid.fromFile(str(tg_path))
        return {name: tiers for name, tiers in zip(t.getNames(), t.tiers)}
    except Exception:
        return {"words": [], "phones": []}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_representation", required=True, type=Path)
    parser.add_argument("--output")
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--textgrid_dir", required=True, type=Path)
    args = parser.parse_args()

    # 1. Chargement et renommage des couches
    all_rows = []
    files = list(args.audio_representation.glob("*.pkl*"))
    for f in tqdm(files, desc="Chargement"):
        with open(f, "rb") as pkl:
            data = pickle.load(pkl)
            # Transformation 'embedding_layer_X' -> 'layer_X'
            new_row = {k.replace("embedding_layer_", "layer_") if k.startswith("embedding_layer_") else k: v for k, v in data.items()}
            all_rows.append(new_row)
    
    df = pd.DataFrame(all_rows)

    # 2. Merge avec metadata (via la colonne 'filename' du CSV)
    df["speaker"] = df["filename"].str.upper()
    metadata = pd.read_csv(args.metadata_path, sep="\t")
    metadata["speaker"] = metadata["filename"].str.upper()
    df = pd.merge(df, metadata, on="speaker")

    # 3. Ajout des TextGrids depuis le dossier recordings/recordings
    print(f"Recherche des TextGrids dans {args.textgrid_dir}...")
    def get_tg(row):
        # On cherche le fichier .TextGrid correspondant à l'identifiant (ex: balanta.TextGrid)
        path = args.textgrid_dir / f"{row['filename_y']}.TextGrid"
        return read_textgrid(path)

    df_tg = df.apply(get_tg, axis=1, result_type="expand")
    df = pd.concat([df, df_tg], axis=1)

    # 4. Reconstruction de la phrase
    df["sentence"] = df["words"].apply(
        lambda x: " ".join(interval.mark for interval in x if interval.mark)
    )
    # remove \t which correspond to input errors by annotators
    df["sentence"] = df["sentence"].apply(lambda x: " ".join(x.split()))

    if args.output:
        df.to_pickle(args.output)