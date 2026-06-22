import pytest

from inference_engine.engine.stage_tracker import StageTracker


def test_prefill_record_stored():
    tracker = StageTracker()
    tracker.record_prefill(2, 200, 15.0, 512)
    assert len(tracker._prefill_records) == 1
    assert tracker._prefill_records[0].budget_utilization == pytest.approx(200 / 512)


def test_decode_record_stored():
    tracker = StageTracker()
    tracker.record_decode(3, 8.0, 8)
    assert len(tracker._decode_records) == 1
    assert tracker._decode_records[0].sequences_decoded == 3


def test_prefill_summary_empty():
    assert all(value == 0 for value in StageTracker().prefill_summary().values())


def test_decode_summary_empty():
    assert all(value == 0 for value in StageTracker().decode_summary().values())


def test_prefill_summary_correct():
    tracker = StageTracker()
    for sequences, tokens, latency in [(1, 100, 10.0), (2, 200, 20.0), (3, 300, 30.0)]:
        tracker.record_prefill(sequences, tokens, latency, 512)
    summary = tracker.prefill_summary()
    assert summary["avg_latency_ms"] == pytest.approx(20.0)
    assert summary["total_tokens_prefilled"] == 600
    assert summary["total_sequences_prefilled"] == 6
    assert summary["avg_budget_utilization"] == pytest.approx(200 / 512)


def test_full_report_structure():
    tracker = StageTracker()
    tracker.record_prefill(1, 10, 2.0, 512)
    tracker.record_decode(1, 1.0, 8)
    report = tracker.full_report()
    assert set(report) == {"prefill", "decode"}
    assert report["prefill"]["total_iterations"] == 1
    assert report["decode"]["total_iterations"] == 1
