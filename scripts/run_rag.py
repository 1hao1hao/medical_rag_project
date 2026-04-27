import argparse
import json
import sys
from pathlib import Path

# 确保在未安装包时也能从源码目录导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.config import load_config
from medagent.pipeline import run_baseline_rag


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Baseline RAG Pipeline")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--question", required=True, help="用户问题")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        result = run_baseline_rag(config=config, question=args.question)
    except Exception as exc:
        print("运行失败: {0}".format(exc), file=sys.stderr)
        return 1

    output = {
        "answer": result["answer"],
        "retrieved_contexts": result["retrieved_contexts"],
        "scores": result["scores"],
        "latency": result["latency"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
