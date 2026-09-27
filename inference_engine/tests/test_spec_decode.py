"""tests/test_spec_decode.py — n-gram (prompt lookup) proposer."""

from __future__ import annotations

from inference_engine.engine.spec_decode import NgramProposer


def test_proposes_continuation_of_most_recent_match():
    p = NgramProposer(ngram_max=3, ngram_min=1)
    tokens = [1, 2, 3, 4, 5, 9, 9, 1, 2, 3, 7, 8, 1, 2, 3]
    # suffix [1, 2, 3] last occurred at index 7 → followed by 7, 8, 1
    assert p.propose(tokens, 3) == [7, 8, 1]


def test_falls_back_to_shorter_ngrams():
    p = NgramProposer(ngram_max=4, ngram_min=2)
    tokens = [5, 6, 7, 8, 0, 0, 6, 7]
    assert p.propose(tokens, 2) == [8, 0]


def test_no_match_returns_empty():
    p = NgramProposer(ngram_max=3, ngram_min=2)
    assert p.propose([1, 2, 3, 4, 5], 4) == []
    assert p.propose([1], 4) == []
    assert p.propose([1, 2, 1, 2], 0) == []


def test_proposal_truncated_at_end_of_history():
    p = NgramProposer(ngram_max=2, ngram_min=2)
    tokens = [1, 2, 3, 1, 2]
    assert p.propose(tokens, 5) == [3, 1, 2]
