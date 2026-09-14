# Joint CTC and contrastive training

These experiments optimize LibriSpeech ASR CTC and L2-ARCTIC prompt-level
SupCon in one training run. Each successful update after warm-up accumulates
one LibriSpeech CTC gradient and one L2-ARCTIC SupCon gradient before a shared
optimizer step.

For the initial Arabic experiment, the L2-ARCTIC training data contain the five
non-Arabic accents. Arabic is used only for final evaluation. LibriSpeech
train-clean-100 supplies CTC supervision, and dev-clean selects the best
checkpoint by WER.

Both variants start with one LibriSpeech epoch in which only the random CTC head
is trained. Afterwards:

- `freeze-18` keeps the convolutional feature encoder and Transformer layers
  1--18 frozen;
- `full-transformer` keeps the convolutional feature encoder frozen and updates
  all 24 Transformer layers.

The two configs otherwise use the same seed, batches, losses, optimizer,
scheduler, update budget, and evaluation frequency.

```bash
sbatch scripts/slurm/run_joint_training.sbatch freeze-18 13 --smoke
sbatch scripts/slurm/run_joint_training.sbatch full-transformer 13 --smoke
```

Submit the two full jobs independently so Slurm can allocate two GPUs:

```bash
sbatch scripts/slurm/run_joint_training.sbatch freeze-18 13
sbatch scripts/slurm/run_joint_training.sbatch full-transformer 13
```

Once each training job has produced `checkpoint_best.pt`, evaluate the held-out
Arabic test fold with:

```bash
sbatch scripts/slurm/run_joint_evaluation.sbatch freeze-18
sbatch scripts/slurm/run_joint_evaluation.sbatch full-transformer
```

The WER files are written below the corresponding variant's `outputs/seed=13/`
directory. This evaluation never uses Arabic examples for optimization or
checkpoint selection.
