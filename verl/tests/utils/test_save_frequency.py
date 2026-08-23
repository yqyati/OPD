import pytest

from verl.utils.config import validate_save_frequency


@pytest.mark.parametrize("value", [-10, -1, 0, None, 1.5, True])
def test_rejects_disabled_or_invalid_save_frequency(value):
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_save_frequency(value)


@pytest.mark.parametrize("value", [1, 484, 1_000_000_000])
def test_accepts_positive_save_frequency(value):
    validate_save_frequency(value)
