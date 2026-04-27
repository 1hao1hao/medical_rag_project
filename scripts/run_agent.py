import argparse
import json
import sys
from pathlib import Path

# 确保在未安装包时也能从源码目录导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.agent import run_agentic_rag
from medagent.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="运行轻量 Agentic RAG 状态机")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--question", required=True, help="用户问题")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        result = run_agentic_rag(config=config, question=args.question)
    except Exception as exc:
        print("运行 Agent 失败: {0}".format(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
