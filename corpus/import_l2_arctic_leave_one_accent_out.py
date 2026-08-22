"""Compatibility entry point for the packaged raw L2-ARCTIC importer.

Usage from the repository root:
  PYTHONPATH=src python corpus/import_l2_arctic_leave_one_accent_out.py \
    --corpus-dir data/raw/l2_arctic/speakers \
    --output-dir data/processed/l2_arctic_leave_one_accent_out
"""

from accented_asr.cli.import_l2_arctic_leave_one_accent_out import main


if __name__ == "__main__":
    main()
