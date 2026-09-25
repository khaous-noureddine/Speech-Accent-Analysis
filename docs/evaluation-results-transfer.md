# Transfer evaluation results from Magi

This procedure collects existing evaluation artifacts without copying model
checkpoints. It includes aggregate metrics, per-utterance predictions, resolved
configurations, logs, and Snakemake benchmarks. The prediction files are useful
for error analysis and paired statistical tests without running inference again.

## 1. Create the archive on Magi

Run from the repository on Magi:

```bash
cd ~/napster/accented-speech-recognition

find experiments -type f \
  \( -path '*/outputs/seed=*/greedy/*' \
     -o -path '*/outputs/seed=*/beam_4gram/*' \) \
  \( -name '*.json' \
     -o -name '*.parquet' \
     -o -name '*.log' \
     -o -name '*.tsv' \) \
  -print0 \
| tar --null -czf ~/eval-results-$(date +%F).tar.gz --files-from=-
```

The archive is written to the Magi home directory and keeps the original
`experiments/...` hierarchy. Restricting paths to the `greedy` and
`beam_4gram` decoder directories prevents Stage 2/Stage 3 training metadata
from being included.

## 2. Check the archive

```bash
archive=~/eval-results-$(date +%F).tar.gz
ls -lh "$archive"
tar -tzf "$archive" | head -n 30
tar -tzf "$archive" | wc -l
```

Summarize the experiment families represented in the archive:

```bash
tar -tzf "$archive" \
  | awk -F/ '$1 == "experiments" {print $2}' \
  | sort | uniq -c
```

Confirm that both decoder types are present when they have been run:

```bash
tar -tzf "$archive" \
  | awk -F/ '{for (i=1; i<=NF; i++) if ($i=="greedy" || $i=="beam_4gram") print $i}' \
  | sort | uniq -c
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
