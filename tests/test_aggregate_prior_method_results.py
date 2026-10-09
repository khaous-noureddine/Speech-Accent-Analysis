import json

from scripts.analysis.aggregate_prior_method_results import collect_rows, macro_rows


def write_metrics(base, method, accent, decoder, dataset, wer):
    path = (
        base
        / method
        / accent
        / "full-transformer"
        / "outputs"
        / "seed=13"
        / decoder
        / dataset
        / "metrics.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "wer": wer,
                "errors": 2,
                "substitutions": 1,
                "deletions": 1,
                "insertions": 0,
                "reference_words": 10,
                "utterances": 1,
                "checkpoint_sha256": accent,
                "smoke": False,
            }
        ),
        encoding="utf-8",
    )


def test_collect_and_macro_average(tmp_path):
    write_metrics(tmp_path, "accent-dat", "arabic", "greedy", "aesrc", 0.2)
    write_metrics(tmp_path, "accent-dat", "chinese", "greedy", "aesrc", 0.4)

    rows = collect_rows(tmp_path)
    assert [row["wer_percent"] for row in rows] == [20.0, 40.0]

    macros = macro_rows(rows)
    assert macros == [
        {
            "method": "accent-dat",
            "decoder": "greedy",
            "dataset": "aesrc",
            "folds": 2,
            "accents": "arabic|chinese",
            "macro_wer_percent": 30.0,
        }
    ]
