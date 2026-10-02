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

- `run_utterance_joint_training_100h.sbatch`
- `run_utterance_joint_evaluation_100h.sbatch`
- `run_utterance_joint_training_960h.sbatch`
- `run_utterance_joint_evaluation_960h.sbatch`

For the 960-hour evaluator, array task `0` is greedy, task `1` is 4-gram, and
task `2` is the fold-independent internal CTC baseline. Joint accent models
normally use tasks `0-1`.

## Word-level SupCon — joint training

- `run_word_joint_training.sbatch`
- `run_word_joint_evaluation.sbatch`

The word training launcher receives, in order: model variant, seed,
LibriSpeech hours, and accented-word hours. The evaluation launcher receives
LibriSpeech hours and accented-word hours; its array tasks `0-1` run greedy and
4-gram decoding respectively.
