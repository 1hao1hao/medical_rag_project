from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.config import Settings
from medagent.generation.llm_client import MockLLMClient, create_llm_client


def _mock_config():
    return Settings(
        data_dir="./data",
        db_dir="./chroma_db",
        output_dir="./outputs/dev_cpu",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-large",
        llm_provider="mock",
        llm_model="mock-medagent",
        retrieval_device="cpu",
        generation_device="cpu",
        top_k_recall=15,
        top_k_rerank=3,
        use_reranker=True,
    )


def test_mock_llm_client_generate() -> None:
    client = MockLLMClient()
    messages = [
        {"role": "system", "content": "你是一个测试助手"},
        {"role": "user", "content": "请根据参考资料回答。"},
    ]
    text = client.generate(messages)
    assert "Mock回答" in text
    assert "请根据参考资料回答" in text


def test_create_llm_client_mock() -> None:
    config = _mock_config()
    client = create_llm_client(config)
    assert isinstance(client, MockLLMClient)
