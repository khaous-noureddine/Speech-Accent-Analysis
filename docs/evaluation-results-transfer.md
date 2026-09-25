# Transfer evaluation results from Magi

This procedure collects existing evaluation artifacts without copying model
checkpoints. It includes aggregate metrics, per-utterance predictions, resolved
configurations, logs, and Snakemake benchmarks. The prediction files are useful
for error analysis and paired statistical tests without running inference again.

## 1. Create the archive on Magi

Run from the repository on Magi:

```bash
cd ~/napster/accented-speech-recognition

find experiments \
  -type f \( \
    -name 'metrics.json' \
    -o -name 'predictions.parquet' \
    -o -name 'config.resolved.json' \
    -o -name 'evaluation.log' \
    -o -name 'benchmark.tsv' \
    -o -name 'aggregate*.json' \
    -o -name 'summary*.json' \
  \) \
  -print0 \
| tar --null -czf ~/eval-results-$(date +%F).tar.gz --files-from=-
```

The archive is written to the Magi home directory and keeps the original
`experiments/...` hierarchy.

## 2. Check the archive

```bash
archive=~/eval-results-$(date +%F).tar.gz
ls -lh "$archive"
tar -tzf "$archive" | head -n 30
tar -tzf "$archive" | wc -l
```

## 3. Download it from the local computer

Leave the Magi SSH session, then run on the local computer:

```bash
scp magi:~/eval-results-$(date +%F).tar.gz ~/Downloads/
```

If the local SSH configuration does not define the `magi` alias, replace it
with the full `user@host` address:

```bash
scp noureddine.khaous@MAGI_HOST:~/eval-results-$(date +%F).tar.gz ~/Downloads/
```

The downloaded archive can then be attached to the Codex conversation for
analysis.

## Optional: inspect locally

```bash
tar -tzf ~/Downloads/eval-results-$(date +%F).tar.gz | less
```

Do not extract the archive over the repository unless restoring results is
intentional, because it can overwrite existing evaluation artifacts.
