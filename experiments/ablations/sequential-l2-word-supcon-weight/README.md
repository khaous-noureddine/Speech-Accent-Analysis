# Sequential word-SupCon weight ablation

This first ablation is intentionally restricted to the Arabic held-out fold and
seed 13. It holds the source checkpoint, word segments, batches, optimizer,
schedule, and word-CTC objective fixed while varying only the word-level
SupCon coefficient: 0.01, 0.03, 0.05, and 0.10.

`CTC_word` from the main sequential experiment is the matched lambda-zero
control. A single word-CTC warm-up checkpoint is hard-linked into all four
branches before adaptation, ensuring identical initialization.

Run on Odette after the main sequential campaign has released both GPUs:

```bash
bash scripts/local/run_sequential_l2_word_lambda_ablation.sh 13 0 1 true
```

Each branch selects its checkpoint by the full-utterance L2-ARCTIC development
WER. The script evaluates greedy and 4-gram decoding, skipping unavailable
corpora rather than failing the whole campaign.

Aggregate the completed results with:

```bash
PYTHONPATH=src pixi run python \
  scripts/analysis/aggregate_sequential_l2_word_lambda_ablation.py
```
