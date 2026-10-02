# LibriSpeech 960-hour Arabic pilot

This pilot tests whether utterance-level SupCon still improves ASR when both
models receive all 960 labelled LibriSpeech hours. Both conditions initialize
from `facebook/wav2vec2-large-lv60`, reserve Arabic entirely from L2-ARCTIC
training, and keep the convolutional feature encoder frozen throughout.

The conditions differ only in the auxiliary objective after the identical
10,000-update CTC-head-only warm-up:

- `ctc-only`: LibriSpeech CTC only (`supcon_weight: 0.0`);
- `joint-supcon`: LibriSpeech CTC plus L2-ARCTIC SupCon (`supcon_weight: 0.1`).

The training corpus combines `train-clean-100`, `train-clean-360`, and
`train-other-500`. The processed parquet points to repository-relative raw FLAC
files and does not duplicate the audio. Following the wav2vec 2.0 fine-tuning
budget for 960 labelled hours, both runs use 320,000 updates. LibriSpeech
`dev-clean` WER selects the best checkpoint. Final evaluation covers held-out Arabic, LibriSpeech
test-clean, AESRC, Speech Accent Archive, and EDACC.

```bash
scripts/download_librispeech_960.sh
sbatch scripts/slurm/run_utterance_joint_training_960h.sbatch ctc-only 13 --smoke
sbatch scripts/slurm/run_utterance_joint_training_960h.sbatch joint-supcon 13 --smoke
sbatch scripts/slurm/run_utterance_joint_training_960h.sbatch ctc-only 13
sbatch scripts/slurm/run_utterance_joint_training_960h.sbatch joint-supcon 13
```
