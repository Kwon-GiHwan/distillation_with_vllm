"""Tests for `app.utils`."""
import json
from app import utils

def test_load_yaml(tmp_path):
    """Verify `load_yaml()` correctly loads configuration."""
    f = tmp_path / "test.yaml"
    f.write_text("key: value")
    assert utils.load_yaml(f) == {"key": "value"}

def test_jsonl_writer(tmp_path):
    """Test `jsonl_writer()` creates valid JSONL output."""
    path = tmp_path / "test.jsonl"
    records = [{"a": 1}, {"b": 2}]
    with utils.jsonl_writer(path) as f:
        for record in records:
            f.write(json.dumps(record).encode("utf-8") + b"\\n")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line for line in f.read().strip().split('\\n') if line]
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

def test_log(capsys):
    """Check `log()` prints to stderr."""
    utils.log("hello", "world")
    captured = capsys.readouterr()
    assert captured.err == "hello world\n"
