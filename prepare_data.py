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

    # 1. Load and rename layers
    all_rows = []
    files = list(args.audio_representation.glob("*.pkl*"))
    for f in tqdm(files, desc="Loading"):
        with open(f, "rb") as pkl:
            data = pickle.load(pkl)
            # Transform 'embedding_layer_X' -> 'layer_X'
            new_row = {k.replace("embedding_layer_", "layer_") if k.startswith("embedding_layer_") else k: v for k, v in data.items()}
            all_rows.append(new_row)
    
    df = pd.DataFrame(all_rows)

    # 2. Merge with metadata (via 'filename' column from CSV)
    df["speaker"] = df["filename"].str.upper()
    metadata = pd.read_csv(args.metadata_path, sep="\t")
    metadata["speaker"] = metadata["filename"].str.upper()
    df = pd.merge(df, metadata, on="speaker")

    # 3. Add TextGrids from recordings/recordings directory
    print(f"Searching for TextGrids in {args.textgrid_dir}...")
    def get_tg(row):
        # Look for the .TextGrid file corresponding to the identifier (e.g., balanta.TextGrid)
        path = args.textgrid_dir / f"{row['filename_y']}.TextGrid"
        return read_textgrid(path)

    df_tg = df.apply(get_tg, axis=1, result_type="expand")
    df = pd.concat([df, df_tg], axis=1)

    # 4. Reconstruct the sentence
    df["sentence"] = df["words"].apply(
        lambda x: " ".join(interval.mark for interval in x if interval.mark)
    )
    # Remove \t which correspond to input errors by annotators
    df["sentence"] = df["sentence"].apply(lambda x: " ".join(x.split()))

    if args.output:
        df.to_pickle(args.output) 