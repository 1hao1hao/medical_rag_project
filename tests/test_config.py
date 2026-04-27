from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.config import Settings, load_config


def test_load_dev_cpu_config() -> None:
    cfg = load_config("configs/dev_cpu.yaml")
    assert isinstance(cfg, Settings)
    assert cfg.data_dir == "./data"
    assert cfg.llm_provider == "mock"
    assert cfg.retrieval_device == "cpu"
    assert cfg.generation_device == "cpu"
    assert cfg.top_k_recall == 15
    assert cfg.top_k_rerank == 3
    assert cfg.use_reranker is True


def test_load_config_invalid_provider(tmpdir) -> None:
    bad = tmpdir.join("bad.yaml")
    bad.write(
        "\n".join(
            [
                "data_dir: ./data",
                "db_dir: ./chroma_db",
                "output_dir: ./outputs",
                "embedding_model: BAAI/bge-m3",
                "reranker_model: BAAI/bge-reranker-large",
                "llm_provider: invalid_provider",
                "llm_model: model",
                "retrieval_device: cpu",
                "generation_device: cpu",
                "top_k_recall: 10",
                "top_k_rerank: 3",
                "use_reranker: true",
            ]
        )
    )

    with pytest.raises(ValueError, match="llm_provider 非法"):
        load_config(str(bad))
