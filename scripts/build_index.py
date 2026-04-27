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
from medagent.ingestion import build_vector_db


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 MedAgent 向量索引")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        result = build_vector_db(config)
    except Exception as exc:
        print("构建索引失败: {0}".format(exc), file=sys.stderr)
        return 1

    print("索引构建成功。")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
