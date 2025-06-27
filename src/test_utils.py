from utils import is_risk_label_valid


def test_valid_labels():
    assert is_risk_label_valid(0)
    assert is_risk_label_valid(1)


def test_invalid_labels():
    assert not is_risk_label_valid(-1)
    assert not is_risk_label_valid(2)
    assert not is_risk_label_valid(None)
