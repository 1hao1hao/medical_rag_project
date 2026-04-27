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


def main() -> None:
    parser = argparse.ArgumentParser(description="打印 MedAgent 配置")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    settings = load_config(args.config)
    print(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
