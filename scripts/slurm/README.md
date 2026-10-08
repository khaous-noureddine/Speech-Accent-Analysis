# Slurm launchers

The launcher names encode the contrastive unit, training topology, action, and
LibriSpeech budget where it matters.

## Utterance-level SupCon — separated training

- `run_utterance_separated_stage2.sbatch`: accent adaptation (Stage 2).
- `run_utterance_separated_stage3.sbatch`: ASR fine-tuning (Stage 3).
- `run_utterance_evaluation_greedy.sbatch`: greedy evaluation for separated
  models and baselines.
- `run_utterance_evaluation_4gram.sbatch`: 4-gram evaluation for separated
  models and baselines.

## Utterance-level SupCon — joint training

- `run_utterance_joint_training.sbatch`: receives the LibriSpeech budget
  (`100h` or `960h`), variant, seed, and optionally one accent.
- `run_utterance_joint_evaluation.sbatch`: receives the LibriSpeech budget,
  accent, and variant.

For the joint evaluator, array task `0` is greedy and task `1` is 4-gram.

## SupCon loss-weight ablation

- `run_gamma_ablation_training.sbatch`: trains the Arabic pilot for gamma
  values `0.01`, `0.05`, `0.1`, `0.5`, and `1.0`.
- `run_gamma_ablation_evaluation.sbatch`: evaluates every gamma with greedy and
  fixed 4-gram decoding.

The training array indices `0-4` follow that gamma order. The evaluation array
uses two consecutive tasks per gamma: greedy, then 4-gram.

## Word-level SupCon — joint training

- `run_word_joint_training.sbatch`
- `run_word_joint_evaluation.sbatch`

The word training launcher receives, in order: model variant, seed,
LibriSpeech hours, and accented-word hours. The evaluation launcher receives
LibriSpeech hours and accented-word hours; its array tasks `0-1` run greedy and
4-gram decoding respectively.

## Prior-method baselines

`run_prior_method_training.sbatch` supports `accent-dat`, `accent-mtl`,
`multidomain-ctc`, and `augmented-view-supcon`. Its array indices `0-5` map to
Arabic, Chinese, Hindi, Korean, Spanish, and Vietnamese. Every fold excludes
the named accent from auxiliary training and uses the same LibriSpeech-100h
CTC setup and checkpoint-selection metric as the main experiment.

```bash
# One method, all six held-out accents (run sequentially).
sbatch --array=0-5%1 scripts/slurm/run_prior_method_training.sbatch accent-dat

# One fold only.
sbatch scripts/slurm/run_prior_method_training.sbatch accent-mtl arabic 13

# Greedy and fixed 4-gram evaluation for one trained fold.
sbatch scripts/slurm/run_prior_method_evaluation.sbatch accent-mtl arabic
```

The augmented-view condition is an utterance-level adaptation of Han et al.'s
augmentation-based contrastive principle, not an exact reproduction of their
character-aligned objective. The multi-domain condition uses L2-ARCTIC
transcriptions and should therefore be reported as a stronger-supervision
baseline.
