"""
Prepare Metadata for Distance-to-Native Analysis
=================================================

Convert Speech Accent Archive metadata to required format.

Required columns:
- speaker: speaker ID (e.g., "ARABIC29", "FRENCH40")
- is_native: boolean (True for native English speakers)
- L1: native language string (e.g., "arabic", "french", "english")
"""

import pandas as pd
import argparse
from pathlib import Path
from loguru import logger


def prepare_metadata(input_tsv, output_csv):
    """
    Convert metadata TSV to required format.
    
    Parameters
    ----------
    input_tsv : Path
        Input metadata file (e.g., mfa_input.tsv)
    output_csv : Path
        Output metadata CSV
    """
    logger.info(f"Loading metadata from {input_tsv}")
    df = pd.read_csv(input_tsv, sep="\t")
    
    # Standardize column names
    df["speaker"] = df["filename"].str.upper()  # Add uppercase speaker ID
    df["native_language"] = df["native_language"].str.lower()  # Normalize language names
    
    # Determine if native English speaker
    df["is_native"] = df["native_language"] == "english"
    
    # Count number of speakers per native language
    lang_counts = df["native_language"].value_counts().to_dict()
    df["count_native_language"] = df["native_language"].map(lang_counts)
    
    logger.info(f"Total speakers: {len(df)}")
    logger.info(f"\nSpeakers per native language:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {lang:15s}: {count:4d} speakers")
    
    # Save
    logger.info(f"Saving to {output_csv}")
    df.to_csv(output_csv, index=False)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Input TSV metadata file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV metadata file")
    parser.add_argument("--max_per_lang", type=int, default=None,
                        help="Maximum number of speakers per language (for balanced sampling)")
    parser.add_argument("--min_per_lang", type=int, default=None,
                        help="Minimum number of speakers per language (filter out rare languages)")
    
    args = parser.parse_args()
    
    meta = prepare_metadata(args.input, args.output)
    
    # Optional: Create balanced subset
    if args.max_per_lang or args.min_per_lang:
        logger.info("\n" + "="*60)
        logger.info("Creating balanced subset...")
        
        balanced = []
        
        for lang in meta["native_language"].unique():
            lang_speakers = meta[meta["native_language"] == lang]
            count = len(lang_speakers)
            
            # Filter by minimum
            if args.min_per_lang and count < args.min_per_lang:
                logger.info(f"Skipping {lang:15s}: only {count} speakers (< {args.min_per_lang})")
                continue
            
            # Sample if too many
            if args.max_per_lang and count > args.max_per_lang:
                lang_speakers = lang_speakers.sample(n=args.max_per_lang, random_state=42)
                logger.info(f"Sampling {lang:15s}: {args.max_per_lang}/{count} speakers")
            else:
                logger.info(f"Keeping  {lang:15s}: {count} speakers")
            
            balanced.append(lang_speakers)
        
        balanced_meta = pd.concat(balanced, axis=0).reset_index(drop=True)
        
        # Save balanced version
        balanced_output = args.output.parent / f"{args.output.stem}_balanced.csv"
        logger.info(f"\nSaving balanced metadata to {balanced_output}")
        logger.info(f"Total speakers in balanced set: {len(balanced_meta)}")
        balanced_meta.to_csv(balanced_output, index=False)
    
    logger.info("✅ Done!")