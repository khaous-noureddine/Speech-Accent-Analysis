# External baselines

This directory contains evaluation-only experiments for published checkpoints.
The model weights are loaded directly from their upstream registry and are
never copied into the repository. Each config pins the upstream revision and
writes its generated artifacts below its own `evaluation/outputs/` directory.

`facebook-wav2vec2-large-960h-lv60` is Meta's official LV-60 pretrained model
fine-tuned on the 960-hour LibriSpeech training set. It is an external
upper-reference system, not the controlled no-Stage-2 baseline, because it uses
substantially more labeled ASR data than our Stage 3 protocol.

`patrickvonplaten-wav2vec2-large-lv60h-100h` is a Transformers checkpoint
fine-tuned from `facebook/wav2vec2-large-lv60` on LibriSpeech train-clean-100.
It therefore matches our backbone and labeled-data amount, while retaining a
different published fine-tuning recipe. The model card reports 4.0 WER on
LibriSpeech test-clean. This is an external 100-hour reference, not Meta's
original Fairseq `wav2vec_vox_100h_new.pt` release.
