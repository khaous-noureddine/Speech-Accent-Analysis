# Sequential L2-ARCTIC adaptation

This experiment starts from the in-house LibriSpeech-100h CTC checkpoint and
compares two strictly paired L2-ARCTIC adaptations:

- `ctc`: L2-ARCTIC CTC only;
- `ctc_supcon`: L2-ARCTIC CTC plus content-parallel SupCon.

The view-construction ablation adds a third branch:

- `ctc_augmented_supcon`: L2-ARCTIC CTC plus SupCon between each recording
  and a waveform-noise/time-mask view of that same recording. All other
  adaptation settings match `ctc_supcon`; recordings by other speakers reading
  the same prompt are negatives in this control.

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

Pass `true` as the fifth argument to evaluate both objectives with greedy and
4-gram decoding immediately after training:

```bash
bash scripts/local/run_sequential_l2_pair.sh arabic 13 0 1 true
```

On Magi/Slurm:

```bash
bash scripts/slurm/submit_sequential_l2_arabic_pilot.sh 13
```

For the matched same-recording-view versus content-parallel ablation, submit
one accent and one seed first:

```bash
bash scripts/slurm/submit_sequential_l2_view_ablation.sh arabic 13
```

After the pilot succeeds, submit all three seeds for an accent:

```bash
bash scripts/slurm/submit_sequential_l2_view_ablation.sh arabic all
```

Submit the complete six-fold, three-seed campaign with:

```bash
bash scripts/slurm/submit_sequential_l2_view_ablation.sh all all
```

The script creates one shared warm-up, launches `ctc_supcon` and
`ctc_augmented_supcon` from that checkpoint, and then evaluates both with
greedy and fixed 4-gram decoding on all configured corpora. The augmented-view
branch requires a second encoder pass for the stochastic view, so wall-clock
cost is higher even though the sampled base recordings and optimization
schedule are matched.

## Three adaptation seeds

All three seeds deliberately start from the same ASR-100h checkpoint so that
the measured variance comes from L2 adaptation rather than ASR pretraining.

```bash
bash scripts/slurm/submit_sequential_l2_three_seeds.sh arabic
```

The submission scripts chain warm-up, both paired adaptations, and greedy/LM
evaluation. Run them from the repository root on Magi. Checkpoints can then be
copied to Odette if evaluation is preferred there.
