"""
This script aims to select recordings from a folder with respect to specific criteria
"""

import argparse
from pathlib import Path
import pandas as pd
import shutil
from loguru import logger

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings_dir", required=True, type=Path)
    parser.add_argument("--metadata_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--native_language", required=True, type=str)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading metadata from {args.metadata_file}...")
    metadata = pd.read_csv(args.metadata_file, sep="\t")
    
    available_languages = set(metadata['native_language'].unique())
    if args.native_language not in available_languages:
        raise ValueError(f"Language '{args.native_language}' not found. Available: {available_languages}")

    selected = metadata[metadata['native_language'] == args.native_language].copy()
    
    # On normalise le nom de base sans extension pour faciliter la recherche multi-format
    selected['stem'] = selected['filename'].apply(lambda x: Path(x).stem)
    # On garde aussi le nom mp3 pour le merge
    selected['mp3_name'] = selected['stem'] + ".mp3"
    
    logger.info(f"Scanning files in {args.recordings_dir}...")
    # On scanne les mp3 pour identifier les dossiers cibles
    mp3_files = list(args.recordings_dir.rglob("*.mp3"))
    df_files = pd.DataFrame({
        "mp3_path": mp3_files,
        "mp3_name": [f.name for f in mp3_files]
    })
    
    final_selection = pd.merge(selected, df_files, on="mp3_name")
    
    if final_selection.empty:
        logger.warning("No files found matching the criteria.")
    else:
        n_speakers = len(final_selection)
        total_copied = 0
        extensions = ['.mp3', '.TextGrid', '.lab']
        
        logger.info(f"Processing {n_speakers} speakers...")
        
        for _, row in final_selection.iterrows():
            mp3_path = Path(row['mp3_path'])
            base_path = mp3_path.parent / mp3_path.stem
            
            for ext in extensions:
                source_file = base_path.with_suffix(ext)
                if source_file.exists():
                    shutil.copy2(source_file, args.output_dir / source_file.name)
                    total_copied += 1
                else:
                    logger.debug(f"Optional file missing: {source_file.name}")
        
        logger.success(f"Task finished. {n_speakers} speakers processed. {total_copied} total files copied to {args.output_dir}")