# External baselines

This directory contains evaluation-only experiments for published checkpoints.
The model weights are loaded directly from their upstream registry and are
never copied into the repository. Each config pins the upstream revision and
writes its generated artifacts below its own `evaluation/outputs/` directory.

`facebook-wav2vec2-large-960h-lv60` is Meta's official LV-60 pretrained model
fine-tuned on the 960-hour LibriSpeech training set. It is an external
upper-reference system, not the controlled no-Stage-2 baseline, because it uses
substantially more labeled ASR data than our Stage 3 protocol.
