# Sequential L2-ARCTIC word-level adaptation

This experiment is the word-level counterpart of
`experiments/sequential-l2-adaptation`. It starts each held-out-accent fold
from the fixed in-house LibriSpeech-100h CTC checkpoint, then runs:

1. one epoch of head-only L2-ARCTIC CTC warm-up on full utterances;
2. full-transformer L2-ARCTIC adaptation with
   `L_CTC^L2 + lambda L_SupCon^word`;
3. checkpoint selection exclusively by full L2-ARCTIC development WER.

The CTC term uses full L2-ARCTIC utterances. The contrastive term uses aligned
word segments from `data/processed/l2_arctic_words/<accent>/corpus.parquet`.
Each word batch contains eight lexical classes and one realization from each of
five seen accents. The held-out accent and its test prompts remain unavailable
to both terms during training and development.

The first run covers all six held-out accents with seed 13 only. It uses a
linear SupCon ramp from 0 to 0.05 during the first 10% of 4,000 adaptation
updates. Logs and TensorBoard events are saved with each objective checkpoint.

On Magi, launch the complete train-and-evaluate pipeline from the repository
root:

```bash
bash scripts/slurm/submit_sequential_l2_word_seed13.sh
```

The script limits every phase to two GPUs. It completes all warm-ups before
adaptation, and all adaptations before evaluation, so the limit holds across
the entire dependency graph.
