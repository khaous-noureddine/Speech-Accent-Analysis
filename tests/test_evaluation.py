import pytest

from accented_asr.evaluation.metrics import (
    NORMALIZATION_VERSION,
    aggregate_edit_counts,
    normalize_for_wer,
    score_utterance,
)


def test_normalization_matches_character_tokenizer_contract():
    assert NORMALIZATION_VERSION == "english_char_v1"
    assert normalize_for_wer("  Don’t, stop!  ") == "DON'T STOP"


def test_normalization_rejects_unexpanded_numbers():
    with pytest.raises(ValueError, match="must be expanded"):
        normalize_for_wer("chapter 42")


def test_utterance_edit_counts_cover_substitution():
    result = score_utterance("THE CAT SAT", "THE DOG SAT")
    assert result["substitutions"] == 1
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["hits"] == 2
    assert result["reference_words"] == 3
    assert result["wer"] == pytest.approx(1 / 3)


def test_empty_hypothesis_counts_all_reference_words_as_deletions():
    result = score_utterance("THE CAT SAT", "")
    assert result["deletions"] == 3
    assert result["errors"] == 3
    assert result["reference_words"] == 3
    assert result["wer"] == 1.0


def test_corpus_wer_sums_counts_instead_of_averaging_utterance_wer():
    records = [
        score_utterance("WRONG", "OTHER"),
        score_utterance("ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE", "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE"),
    ]
    result = aggregate_edit_counts(records)
    assert result["utterances"] == 2
    assert result["errors"] == 1
    assert result["reference_words"] == 10
    assert result["wer"] == 0.1


def test_empty_reference_is_a_data_contract_error():
    with pytest.raises(ValueError, match="reference is empty"):
        score_utterance("...", "HELLO")

