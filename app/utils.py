from pathlib import Path
import orjson, yaml, contextlib, sys

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@contextlib.contextmanager
def jsonl_writer(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        yield f

def log(*args):
    print(*args, file=sys.stderr)
