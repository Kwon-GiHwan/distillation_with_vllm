import pytest, os, sys
from unittest.mock import MagicMock

# Unset distributed environment variables
for v in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
    os.environ.pop(v, None)

# Mock heavyweight modules to avoid installation during testing
sys.modules["vllm"] = MagicMock()
sys.modules["evaluate"] = MagicMock()
sys.modules["bert_score"] = MagicMock()

os.environ["CUDA_VISIBLE_DEVICES"] = ""

@pytest.fixture(scope="session", autouse=True)
def setup_dirs(tmp_path_factory):
    os.makedirs("data/distilled", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
