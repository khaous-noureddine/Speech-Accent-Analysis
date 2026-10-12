# Sequential L2-ARCTIC word-level adaptation

This experiment is the word-level counterpart of
`experiments/sequential-l2-adaptation`. It starts each held-out-accent fold
from the fixed in-house LibriSpeech-100h CTC checkpoint, then runs:

1. one epoch of head-only L2-ARCTIC CTC warm-up on aligned word segments;
2. full-transformer L2-ARCTIC adaptation with
   `L_CTC^word + lambda L_SupCon^word` on the same word batches;
3. checkpoint selection exclusively by full L2-ARCTIC development WER.

Both training losses use only aligned word segments from
`data/processed/l2_arctic_words/<accent>/corpus.parquet`; full L2-ARCTIC
utterances are used only for development-based checkpoint selection and final
evaluation. Each word batch contains eight lexical classes and one realization
from each of five seen accents. The held-out accent and its test prompts remain
unavailable to both training losses.

The first run covers all six held-out accents with seed 13 only. It uses a
linear SupCon ramp from 0 to 0.05 during the first 10% of 4,000 adaptation
updates. It trains both the word-CTC control and the word-CTC plus word-SupCon
model. Logs and TensorBoard events are saved with each objective checkpoint.

On Magi, launch the complete train-and-evaluate pipeline from the repository
root:

```bash
bash scripts/slurm/submit_sequential_l2_word_seed13.sh
```

If the six word manifests are absent, the launcher first submits their
CPU-only construction from raw L2-ARCTIC TextGrids. It then chains warm-up,
adaptation, and evaluation with `afterok` dependencies. The script limits every
GPU phase to two concurrent jobs, completes all warm-ups before adaptation,
and completes all adaptations before evaluation.
