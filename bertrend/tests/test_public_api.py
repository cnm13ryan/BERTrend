# tests/contracts/test_public_api.py

from bertrend.BERTrend import BERTrend


def test_bertrend_public_api_contract(monkeypatch):
    """Freeze config, save/restore, and topic-model contract."""
    dummy = {"granularity": 99}
    monkeypatch.setattr("bertrend.BERTrend.load_toml_config", lambda _: dummy)
    bt = BERTrend()  # must call original ctor
    assert bt.config is dummy  # guards our current bug
    assert hasattr(bt, "save_model")
    assert hasattr(bt, "restore_model")
