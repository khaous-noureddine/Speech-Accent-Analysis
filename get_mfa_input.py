import pandas as pd
import argparse
from pathlib import Path
from loguru import logger
import os

# def create_lab_files(audio_dir, text):
#     path = Path(audio_dir)

#     for audio_file in path.glob("*.mp3"):
#         lab_path = audio_file.with_suffix(".lab")
#         with open(lab_path, "w", encoding="utf-8") as f:
#             f.write(text)
#         print(f"Created : {lab_path.name}")


def create_lab_files(audio_dir, text, lab_dir=None):
    audio_dir = Path(audio_dir)

    if lab_dir is not None:
        lab_dir = Path(lab_dir)
        lab_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in audio_dir.glob("*.mp3"):
        if lab_dir is None:
            lab_path = audio_file.with_suffix(".lab")   # même dossier que mp3
        else:
            lab_path = lab_dir / (audio_file.stem + ".lab")  # dossier choisi

        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Created : {lab_path}")
        
        
def create_mfa_input(csv_path, output_tsv):
    text = "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."

    df = pd.read_csv(csv_path)
    df_ready = df[df['file_missing?'] == False].copy()
    df_ready['wav'] = df_ready['filename'].astype(str) + ".mp3"
    df_ready['sentence'] = text
    df_ready.to_csv(output_tsv, sep='\t', index=False)
    print(f"File {output_tsv} created succesfully with {len(df_ready)} entries.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--lab_dir", required=False, type=Path)
    args = parser.parse_args()

    
    phrases = [
    # "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob."
    "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
    ]
    text = "\n".join(phrases).strip()
    logger.info("Creating .lab files for MFA")
    create_lab_files(args.corpus_dir, text, lab_dir=args.lab_dir)
    
    logger.info("Creating MFA input TSV file")
    create_mfa_input(args.data_dir / "speakers_all.csv", args.data_dir / "mfa_input.tsv")