# Reviewer control experiments

## Shuffled content labels

The control uses the same L2-ARCTIC fold, prompt sampler, batch size, audio, and
number of positives as utterance CP-SupCon. Within every batch, the examples of
each true prompt are assigned to distinct balanced pseudo-groups. This changes
the positive pairs rather than applying a bijective renaming of prompt IDs.
The pseudo-group RNG is local and step-seeded, so it does not perturb the model
dropout or data-sampling RNG streams.

```bash
shuffle_job=$(sbatch --parsable --array=0-5%1 \
  scripts/slurm/run_shuffled_supcon_training.sbatch)
sbatch --dependency="afterok:${shuffle_job}" --array=0-11%1 \
  scripts/slurm/run_shuffled_supcon_evaluation.sbatch
```

## LibriSpeech test-other

Download the official OpenSLR subset once. Snakemake creates the canonical
Parquet automatically when the first evaluation starts.

```bash
bash scripts/download_librispeech_test_other.sh
sbatch --array=0-1%1 scripts/slurm/run_test_other_evaluation.sbatch asr-only
sbatch --array=0-11%1 scripts/slurm/run_test_other_evaluation.sbatch md-ft
sbatch --array=0-11%1 scripts/slurm/run_test_other_evaluation.sbatch cp-supcon-utterance
sbatch --array=0-1%1 scripts/slurm/run_test_other_evaluation.sbatch cp-supcon-word
```

The combined families use the same launcher after their checkpoints finish:

```bash
sbatch --array=0-11%1 scripts/slurm/run_test_other_evaluation.sbatch combined-utterance
sbatch --array=0-1%1 scripts/slurm/run_test_other_evaluation.sbatch combined-word
```

## Cross-accent L2-ARCTIC geometry

For each fold, queries are all four speakers from the held-out accent reading
the 95 test prompts. The gallery contains the dev speaker from each of the five
seen accents reading the same prompts. These seen-accent/test-prompt crossings
are recovered from the canonical inventory; they were deliberately omitted
from the training fold. Metrics are reported for raw unit vectors and vectors
centered with the seen-accent gallery mean.

The same extraction runs a speaker-disjoint accent probe. For every accent,
two fixed speakers train the probe and the remaining two test it. Only 20 test
prompts are used for the probe to control inference cost.

Run the six folds after the combined checkpoints are complete:

```bash
analysis_job=$(sbatch --parsable --array=0-5%1 \
  scripts/slurm/run_cross_accent_analysis.sbatch)
```

Aggregate after the array succeeds:

```bash
sbatch --dependency="afterok:${analysis_job}" --partition=SMP-256c \
  --account=efl --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --wrap='pixi run python scripts/analysis/aggregate_cross_accent_results.py \
    --input-root experiments/analysis/cross-accent-l2-arctic/librispeech-100h/outputs/seed=13'
```
