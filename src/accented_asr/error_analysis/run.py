"""Compare ASR predictions before and after SupCon at sentence and word levels."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from accented_asr.error_analysis.alignment import error_transitions
from accented_asr.error_analysis.analysis import (
    INTRODUCED_TRANSITIONS,
    RESOLVED_TRANSITIONS,
    sentence_category,
    summarize_comparison,
    top_word_counts,
)


REQUIRED_COLUMNS = {
    "utterance_id", "speaker_id", "reference_normalized", "hypothesis_normalized",
    "reference_words", "errors", "substitutions", "deletions", "insertions",
}


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_commit(root: Path) -> str | None:
    """Return the current revision when Git is available on the compute node."""

    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    result = subprocess.run(
        [git_executable, "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_config(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(document) != {"error_analysis"}:
        raise ValueError("Config requires exactly one error_analysis section.")
    config = document["error_analysis"]
    required = {
        "baseline", "comparisons", "datasets", "decoders", "matching_keys",
        "bootstrap_replicates", "examples_per_category", "output_dir",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing error-analysis fields: {sorted(missing)}")
    return config


def prediction_path(
    root: Path, model: dict[str, Any], decoder: str, dataset_directory: str
) -> Path:
    return resolve(root, model["output_dir"]) / f"seed={model['seed']}" / decoder / dataset_directory / "predictions.parquet"


def read_predictions(path: Path, keys: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = (REQUIRED_COLUMNS | set(keys)) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks prediction columns: {sorted(missing)}")
    if frame.duplicated(keys).any():
        duplicates = frame.loc[frame.duplicated(keys, keep=False), keys].head().to_dict("records")
        raise ValueError(f"Non-unique matching keys in {path}: {duplicates}")
    return frame


def compare_predictions(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    keys: list[str],
    comparison: str,
    dataset: str,
    decoder: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_candidates = (
        "accent", "native_language", "gender", "audio_path", "duration_s",
        "transcript", "dataset", "split",
    )
    metadata = [column for column in metadata_candidates if column in before.columns]
    before_columns = keys + metadata + [
        "reference_normalized", "hypothesis_normalized", "reference_words",
        "errors", "substitutions", "deletions", "insertions",
    ]
    after_columns = keys + [
        "reference_normalized", "hypothesis_normalized", "reference_words",
        "errors", "substitutions", "deletions", "insertions",
    ]
    merged = before[before_columns].merge(
        after[after_columns], on=keys, how="outer", suffixes=("_before", "_after"),
        indicator=True, validate="one_to_one",
    )
    unmatched = merged.loc[merged["_merge"] != "both", keys + ["_merge"]]
    if not unmatched.empty:
        raise ValueError(
            f"{comparison}/{decoder}/{dataset} has {len(unmatched)} unmatched utterances: "
            f"{unmatched.head().to_dict('records')}"
        )
    if not (
        merged["reference_normalized_before"] == merged["reference_normalized_after"]
    ).all():
        raise ValueError(f"Reference mismatch for {comparison}/{decoder}/{dataset}.")
    if not (merged["reference_words_before"] == merged["reference_words_after"]).all():
        raise ValueError(f"Reference word-count mismatch for {comparison}/{decoder}/{dataset}.")

    sentence_rows, transition_rows = [], []
    for row in merged.to_dict("records"):
        identity = {key: row[key] for key in keys}
        common = {
            "comparison": comparison,
            "dataset_name": dataset,
            "decoder": decoder,
            **identity,
        }
        for column in metadata:
            common[column] = row[column]
        before_errors, after_errors = int(row["errors_before"]), int(row["errors_after"])
        before_hypothesis = str(row["hypothesis_normalized_before"])
        after_hypothesis = str(row["hypothesis_normalized_after"])
        sentence_rows.append(
            {
                **common,
                "reference": str(row["reference_normalized_before"]),
                "before_hypothesis": before_hypothesis,
                "after_hypothesis": after_hypothesis,
                "reference_words": int(row["reference_words_before"]),
                "before_errors": before_errors,
                "after_errors": after_errors,
                "error_delta": after_errors - before_errors,
                "before_wer": before_errors / int(row["reference_words_before"]),
                "after_wer": after_errors / int(row["reference_words_before"]),
                "before_substitutions": int(row["substitutions_before"]),
                "after_substitutions": int(row["substitutions_after"]),
                "before_deletions": int(row["deletions_before"]),
                "after_deletions": int(row["deletions_after"]),
                "before_insertions": int(row["insertions_before"]),
                "after_insertions": int(row["insertions_after"]),
                "sentence_category": sentence_category(
                    before_errors, after_errors, before_hypothesis, after_hypothesis
                ),
            }
        )
        for transition in error_transitions(
            str(row["reference_normalized_before"]), before_hypothesis, after_hypothesis
        ):
            transition_rows.append({**common, **transition})
    return pd.DataFrame(sentence_rows), pd.DataFrame(transition_rows)


def select_examples(frame: pd.DataFrame, *, per_category: int, seed: int) -> pd.DataFrame:
    selected = []
    group_columns = ["comparison", "decoder", "dataset_name", "sentence_category"]
    for _, group in frame.groupby(group_columns, sort=True):
        group = group.copy()
        if group.iloc[0]["sentence_category"] in {"improved", "fully_corrected"}:
            group = group.sort_values(["error_delta", "utterance_id"])
        elif group.iloc[0]["sentence_category"] in {"degraded", "introduced_error"}:
            group = group.sort_values(["error_delta", "utterance_id"], ascending=[False, True])
        else:
            group = group.sample(frac=1, random_state=seed)
        chosen = group.head(per_category).copy()
        chosen["selection_rule"] = (
            "largest_change" if group.iloc[0]["sentence_category"]
            in {"improved", "fully_corrected", "degraded", "introduced_error"}
            else "seeded_sample"
        )
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def markdown_examples(frame: pd.DataFrame) -> str:
    lines = [
        "# Before/after qualitative ASR examples", "",
        "An example is identified by (speaker_id, utterance_id). "
        "The same scripted prompt can occur for several speakers.", "",
    ]
    for row in frame.to_dict("records"):
        lines.extend(
            [
                f"## {row['comparison']} · {row['decoder']} · {row['dataset_name']} · {row['sentence_category']}",
                "",
                f"- Utterance: `{row['utterance_id']}`",
                f"- Speaker: `{row.get('speaker_id', 'unknown')}`",
                *([f"- Audio: `{row['audio_path']}`"] if row.get("audio_path") else []),
                f"- Selection: `{row['selection_rule']}`",
                f"- Error count: {row['before_errors']} → {row['after_errors']}",
                f"- Reference: {row['reference']}",
                f"- Before (CTC-only): {row['before_hypothesis']}",
                f"- After (SupCon): {row['after_hypothesis']}",
                "",
            ]
        )
    return "\n".join(lines)


def latex_summary(summary: pd.DataFrame) -> str:
    rows = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Comparison & Decoder & Dataset & Before WER & After WER & $\Delta$ WER & Corr. rate \\",
        r"\midrule",
    ]
    for record in summary.to_dict("records"):
        dataset = str(record["dataset"]).replace("_", r"\_")
        comparison = str(record["comparison"]).replace("_", r"\_")
        rows.append(
            f"{comparison} & {record['decoder']} & {dataset} & "
            f"{record['before_wer_percent']:.2f} & {record['after_wer_percent']:.2f} & "
            f"{record['wer_delta_points']:+.2f} & {record['correction_rate_percent']:.1f} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    keys = [str(key) for key in config["matching_keys"]]
    baseline = config["baseline"]
    seed = int(config.get("seed", 13))
    all_sentences, all_transitions = [], []
    summaries: dict[str, Any] = {}
    flat_summaries = []

    for comparison_name, after_model in config["comparisons"].items():
        summaries[comparison_name] = {}
        for decoder in config["decoders"]:
            summaries[comparison_name][decoder] = {}
            for dataset, directories in config["datasets"].items():
                before_path = prediction_path(root, baseline, decoder, directories["baseline"])
                after_path = prediction_path(root, after_model, decoder, directories[comparison_name])
                before = read_predictions(before_path, keys)
                after = read_predictions(after_path, keys)
                sentences, transitions = compare_predictions(
                    before, after, keys=keys, comparison=comparison_name,
                    dataset=dataset, decoder=decoder,
                )
                result = summarize_comparison(
                    sentences,
                    transitions,
                    seed=seed,
                    bootstrap_replicates=int(config["bootstrap_replicates"]),
                )
                result["before_predictions"] = str(before_path.relative_to(root))
                result["after_predictions"] = str(after_path.relative_to(root))
                summaries[comparison_name][decoder][dataset] = result
                flat_summaries.append(
                    {
                        "comparison": comparison_name,
                        "decoder": decoder,
                        "dataset": dataset,
                        **{key: value for key, value in result.items() if not isinstance(value, dict)},
                        "bootstrap_delta_lower_95": result["paired_bootstrap_wer_delta_points_95"]["lower_95"],
                        "bootstrap_delta_upper_95": result["paired_bootstrap_wer_delta_points_95"]["upper_95"],
                    }
                )
                all_sentences.append(sentences)
                all_transitions.append(transitions)

    sentence_frame = pd.concat(all_sentences, ignore_index=True)
    transition_frame = pd.concat(all_transitions, ignore_index=True)
    summary_frame = pd.DataFrame(flat_summaries)
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sentence_frame.to_parquet(output_dir / "sentence_comparisons.parquet", index=False)
    transition_frame.to_parquet(output_dir / "word_error_transitions.parquet", index=False)
    transition_frame.loc[
        transition_frame["transition"].isin(RESOLVED_TRANSITIONS)
    ].to_csv(output_dir / "corrected_errors.csv", index=False)
    transition_frame.loc[
        transition_frame["transition"].isin(INTRODUCED_TRANSITIONS)
    ].to_csv(output_dir / "introduced_errors.csv", index=False)
    transition_frame.loc[
        transition_frame["transition"].str.startswith("persistent_")
        | transition_frame["transition"].eq("changed_error")
    ].to_csv(output_dir / "persistent_errors.csv", index=False)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    json_dump(output_dir / "summary.json", summaries)

    word_records = []
    for group_key, group in transition_frame.groupby(
        ["comparison", "decoder", "dataset_name"], sort=True
    ):
        for record in top_word_counts(
            group,
            RESOLVED_TRANSITIONS | INTRODUCED_TRANSITIONS
            | {"persistent_substitution", "persistent_deletion", "persistent_insertion", "changed_error"},
            limit=int(config.get("top_words", 30)),
        ):
            word_records.append(
                {"comparison": group_key[0], "decoder": group_key[1], "dataset": group_key[2], **record}
            )
    pd.DataFrame(word_records).to_csv(output_dir / "word_transition_counts.csv", index=False)
    confusion_source = transition_frame.loc[
        transition_frame["before_status"].eq("substitution")
        | transition_frame["after_status"].eq("substitution")
    ].copy()
    confusion_source[
        ["comparison", "decoder", "dataset_name", "transition", "reference_word", "before_word", "after_word"]
    ].value_counts(dropna=False).rename("count").reset_index().to_csv(
        output_dir / "substitution_confusions.csv", index=False
    )

    examples = select_examples(
        sentence_frame,
        per_category=int(config["examples_per_category"]),
        seed=seed,
    )
    examples.to_parquet(output_dir / "qualitative_examples.parquet", index=False)
    (output_dir / "qualitative_examples.md").write_text(
        markdown_examples(examples), encoding="utf-8"
    )
    (output_dir / "paper_table.tex").write_text(
        latex_summary(summary_frame), encoding="utf-8"
    )
    json_dump(
        output_dir / "run_metadata.json",
        {
            "config": str(config_path.relative_to(root)),
            "git_commit": git_commit(root),
            "seed": seed,
            "comparisons": list(config["comparisons"]),
            "decoders": list(config["decoders"]),
            "datasets": list(config["datasets"]),
        },
    )
    (output_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
