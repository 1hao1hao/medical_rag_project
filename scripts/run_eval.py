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
from medagent.evaluation.runner import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 MedAgent 评测 Harness")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["baseline", "agentic"],
        help="评测使用的 pipeline 类型",
    )
    parser.add_argument("--dataset", required=True, help="JSONL 评测数据路径")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        summary = run_evaluation(
            config=config,
            pipeline=args.pipeline,
            dataset_path=args.dataset,
            output_dir="outputs",
        )
    except Exception as exc:
        print("评测执行失败: {0}".format(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
