# Sequential L2-ARCTIC adaptation

This experiment starts from the in-house LibriSpeech-100h CTC checkpoint and
compares two strictly paired L2-ARCTIC adaptations:

- `ctc`: L2-ARCTIC CTC only;
- `ctc_supcon`: L2-ARCTIC CTC plus content-parallel SupCon.

Each fold retains the existing leakage-safe split: the held-out accent and test
prompts are absent from training and development. Checkpoints are selected only
by full L2-ARCTIC development WER. Both branches load the exact same one-epoch
head-only warm-up checkpoint.

The adaptation uses 40 epochs of 100 balanced prompt batches (4,000 updates),
with 8 prompts and 5 speakers per prompt. The SupCon weight ramps linearly from
0 to 0.05 during the first 10% of updates. The convolutional feature encoder
stays frozen; the Transformer and CTC head are fine-tuned. Training stops after
five consecutive epochs without an improvement in full L2 development WER.

## Arabic pilot

On Odette, use both local GPUs after the shared warm-up:

```bash
nohup bash scripts/local/run_sequential_l2_pair.sh arabic 13 0 1 \
  > outputs/sequential-l2-adaptation/arabic-launch.log 2>&1 &
```

On Magi/Slurm:

```bash
bash scripts/slurm/submit_sequential_l2_arabic_pilot.sh 13
```

## Three adaptation seeds

All three seeds deliberately start from the same ASR-100h checkpoint so that
the measured variance comes from L2 adaptation rather than ASR pretraining.

```bash
bash scripts/slurm/submit_sequential_l2_three_seeds.sh arabic
```

The submission scripts chain warm-up, both paired adaptations, and greedy/LM
evaluation. Run them from the repository root on Magi. Checkpoints can then be
copied to Odette if evaluation is preferred there.
