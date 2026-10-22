from app.scoring import priority_for


def test_priority_cutoffs():
    assert priority_for(70) == "High"
    assert priority_for(40) == "Medium"
    assert priority_for(39.99) == "Low"


def test_score_never_exceeds_range():
    for score in (0, 0.01, 50, 99.99, 100):
        assert 0 <= score <= 100
