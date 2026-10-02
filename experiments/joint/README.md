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
sbatch scripts/slurm/run_utterance_joint_training_100h.sbatch freeze-18 13 --smoke
sbatch scripts/slurm/run_utterance_joint_training_100h.sbatch full-transformer 13 --smoke
```

Submit the two full jobs independently so Slurm can allocate two GPUs:

```bash
sbatch scripts/slurm/run_utterance_joint_training_100h.sbatch freeze-18 13
sbatch scripts/slurm/run_utterance_joint_training_100h.sbatch full-transformer 13
```

After the Arabic pilot, submit the five remaining accents sequentially within
each variant. The two arrays may run concurrently, so at most two GPUs are used:

```bash
sbatch --array=1-5%1 scripts/slurm/run_utterance_joint_training_100h.sbatch freeze-18 13
sbatch --array=1-5%1 scripts/slurm/run_utterance_joint_training_100h.sbatch full-transformer 13
```

Once each training job has produced `checkpoint_best.pt`, evaluate the held-out
L2-ARCTIC fold, LibriSpeech test-clean, AESRC, Speech Accent Archive, and EDACC.
The following command evaluates both variants and all six accents sequentially
inside one Slurm task, so the campaign occupies only one GPU:

```bash
sbatch scripts/slurm/run_utterance_joint_evaluation_100h.sbatch all all
```

To evaluate only one variant, or to skip models whose training has not finished:

```bash
sbatch scripts/slurm/run_utterance_joint_evaluation_100h.sbatch freeze-18 all
sbatch scripts/slurm/run_utterance_joint_evaluation_100h.sbatch all all --skip-missing
```

The script checks all requested checkpoints before taking the GPU. Without
`--skip-missing`, one absent checkpoint stops the campaign before inference.
Existing complete Snakemake outputs are reused. Per-dataset WER files and
`metrics_summary.json` are written below each model's
`outputs/seed=13/greedy/` directory.
