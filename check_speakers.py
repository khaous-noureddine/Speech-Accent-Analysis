"""
analyse_arctic_data.py

Analyze the raw CMU Arctic dataset (without using the processed parquet)
to verify corpus integrity and compare against the processing script.

Expected structure:
<corpus_dir>/
    <speaker_id>/                  e.g. cmu_us_awb_arctic
        wav/
            arctic_a0001.wav
            ...
        etc/
            txt.done.data

Usage:
python check_speakers.py --speakers_root data/raw/l2_arctic/speakers
"""



import argparse
from pathlib import Path


def find_files_recursive(base: Path, extensions: tuple[str, ...]) -> list[Path]:
    files = []
    for ext in extensions:
        files.extend(base.rglob(f"*{ext}"))
    return sorted(files)


def collect_speaker_stats(speaker_dir: Path) -> dict:
    # Cherche tous les wav et txt sous ce speaker
    wav_files = find_files_recursive(speaker_dir, (".wav", ".WAV"))
    txt_files = find_files_recursive(speaker_dir, (".txt", ".TXT"))

    wav_ids = {p.stem for p in wav_files}
    txt_ids = {p.stem for p in txt_files}

    common_ids = wav_ids & txt_ids
    wav_only = wav_ids - txt_ids
    txt_only = txt_ids - wav_ids

    return {
        "speaker_id": speaker_dir.name,
        "n_wav": len(wav_files),
        "n_txt": len(txt_files),
        "n_common": len(common_ids),
        "n_wav_only": len(wav_only),
        "n_txt_only": len(txt_only),
        "wav_only_examples": sorted(list(wav_only))[:10],
        "txt_only_examples": sorted(list(txt_only))[:10],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compte les wav et transcripts par speaker."
    )
    parser.add_argument(
        "--speakers_root",
        required=True,
        help="Chemin vers le dossier qui contient les dossiers speakers",
    )
    args = parser.parse_args()

    speakers_root = Path(args.speakers_root)

    if not speakers_root.exists():
        raise FileNotFoundError(f"Dossier introuvable: {speakers_root}")
    if not speakers_root.is_dir():
        raise NotADirectoryError(f"Ce n'est pas un dossier: {speakers_root}")

    speaker_dirs = sorted([d for d in speakers_root.iterdir() if d.is_dir()])
    if not speaker_dirs:
        raise ValueError(f"Aucun dossier speaker trouvé dans {speakers_root}")

    print(f"Root: {speakers_root}")
    print(f"Speakers trouvés: {len(speaker_dirs)}\n")

    all_stats = []
    for speaker_dir in speaker_dirs:
        stats = collect_speaker_stats(speaker_dir)
        all_stats.append(stats)

        print(f"[{stats['speaker_id']}]")
        print(f"  wav files        : {stats['n_wav']}")
        print(f"  transcript files : {stats['n_txt']}")
        print(f"  common IDs       : {stats['n_common']}")
        print(f"  wav without txt  : {stats['n_wav_only']}")
        print(f"  txt without wav  : {stats['n_txt_only']}")

        if stats["wav_only_examples"]:
            print(f"  ex wav_only      : {stats['wav_only_examples']}")
        if stats["txt_only_examples"]:
            print(f"  ex txt_only      : {stats['txt_only_examples']}")
        print()

    # Résumé global
    print("=== SUMMARY ===")
    for stats in all_stats:
        print(
            f"{stats['speaker_id']}: "
            f"wav={stats['n_wav']}, "
            f"txt={stats['n_txt']}, "
            f"common={stats['n_common']}, "
            f"wav_only={stats['n_wav_only']}, "
            f"txt_only={stats['n_txt_only']}"
        )


if __name__ == "__main__":
    main()