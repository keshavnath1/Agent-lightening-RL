import pytest

from ml_tools.sql_safety import validate_sql


def test_sql_safety_wrapper_allows_readonly_select():
    result = validate_sql("select 1 limit 10")
    assert result.safe_sql.lower().startswith("select")
    assert result.max_rows <= 1000


def test_sql_safety_wrapper_blocks_delete():
    with pytest.raises(ValueError):
        validate_sql("delete from synthetic_dataset_rows")


def test_sql_safety_wrapper_blocks_ml_data_schema():
    with pytest.raises(ValueError, match="ml_data"):
        validate_sql("select target from ml_data.openml_31_german_credit")
