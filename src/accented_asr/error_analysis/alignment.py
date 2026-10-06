"""Deterministic word alignment and before/after transition labelling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceToken:
    position: int
    reference_word: str
    status: str
    hypothesis_word: str | None


@dataclass(frozen=True)
class Alignment:
    reference_tokens: tuple[ReferenceToken, ...]
    insertions: dict[int, tuple[str, ...]]


def align_words(reference: str, hypothesis: str) -> Alignment:
    """Levenshtein-align normalized word strings with stable tie breaking."""

    ref, hyp = reference.split(), hypothesis.split()
    n, m = len(ref), len(hyp)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0], back[i][0] = i, "deletion"
    for j in range(1, m + 1):
        costs[0][j], back[0][j] = j, "insertion"
    priority = {"correct": 0, "substitution": 1, "deletion": 2, "insertion": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal_status = "correct" if ref[i - 1] == hyp[j - 1] else "substitution"
            candidates = (
                (costs[i - 1][j - 1] + (diagonal_status != "correct"), diagonal_status),
                (costs[i - 1][j] + 1, "deletion"),
                (costs[i][j - 1] + 1, "insertion"),
            )
            costs[i][j], back[i][j] = min(
                candidates, key=lambda item: (item[0], priority[item[1]])
            )

    aligned: list[ReferenceToken] = []
    insertions: dict[int, list[str]] = {}
    i, j = n, m
    while i or j:
        operation = back[i][j]
        if operation in {"correct", "substitution"}:
            aligned.append(ReferenceToken(i - 1, ref[i - 1], operation, hyp[j - 1]))
            i, j = i - 1, j - 1
        elif operation == "deletion":
            aligned.append(ReferenceToken(i - 1, ref[i - 1], "deletion", None))
            i -= 1
        elif operation == "insertion":
            insertions.setdefault(i, []).append(hyp[j - 1])
            j -= 1
        else:
            raise RuntimeError(f"Invalid alignment state at {(i, j)}")
    aligned.reverse()
    normalized_insertions = {
        boundary: tuple(reversed(words)) for boundary, words in insertions.items()
    }
    return Alignment(tuple(aligned), normalized_insertions)


def error_transitions(
    reference: str, before_hypothesis: str, after_hypothesis: str
) -> list[dict[str, object]]:
    """Describe every reference-token and insertion transition before to after."""

    before = align_words(reference, before_hypothesis)
    after = align_words(reference, after_hypothesis)
    if len(before.reference_tokens) != len(after.reference_tokens):
        raise RuntimeError("Alignments do not cover the same reference tokens.")
    transitions: list[dict[str, object]] = []
    for old, new in zip(before.reference_tokens, after.reference_tokens):
        if old.status == "correct" and new.status == "correct":
            transition = "stable_correct"
        elif old.status != "correct" and new.status == "correct":
            transition = f"{old.status}_to_correct"
        elif old.status == "correct" and new.status != "correct":
            transition = f"correct_to_{new.status}"
        elif old.status == new.status and old.hypothesis_word == new.hypothesis_word:
            transition = f"persistent_{old.status}"
        else:
            transition = "changed_error"
        transitions.append(
            {
                "unit": "reference_token",
                "position": old.position,
                "reference_word": old.reference_word,
                "before_status": old.status,
                "before_word": old.hypothesis_word,
                "after_status": new.status,
                "after_word": new.hypothesis_word,
                "transition": transition,
            }
        )

    for boundary in sorted(set(before.insertions) | set(after.insertions)):
        old_words = list(before.insertions.get(boundary, ()))
        new_words = list(after.insertions.get(boundary, ()))
        common = []
        for word in list(old_words):
            if word in new_words:
                old_words.remove(word)
                new_words.remove(word)
                common.append(word)
        for word in common:
            transitions.append(
                {
                    "unit": "insertion",
                    "position": boundary,
                    "reference_word": None,
                    "before_status": "insertion",
                    "before_word": word,
                    "after_status": "insertion",
                    "after_word": word,
                    "transition": "persistent_insertion",
                }
            )
        for word in old_words:
            transitions.append(
                {
                    "unit": "insertion",
                    "position": boundary,
                    "reference_word": None,
                    "before_status": "insertion",
                    "before_word": word,
                    "after_status": "none",
                    "after_word": None,
                    "transition": "insertion_removed",
                }
            )
        for word in new_words:
            transitions.append(
                {
                    "unit": "insertion",
                    "position": boundary,
                    "reference_word": None,
                    "before_status": "none",
                    "before_word": None,
                    "after_status": "insertion",
                    "after_word": word,
                    "transition": "insertion_introduced",
                }
            )
    return transitions
