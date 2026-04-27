import argparse
import sys
from pathlib import Path

# 确保在未安装包时也能从源码目录导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medagent.config import load_config
from medagent.retrieval import RetrievalPipeline


def _print_docs(title: str, docs) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if not docs:
        print("无结果。")
        return

    for idx, doc in enumerate(docs, 1):
        snippet = doc.content.replace("\n", " ").strip()
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        print(
            "[{0}] score={1:.4f} source={2}".format(
                idx,
                doc.score,
                doc.source,
            )
        )
        print("内容: {0}".format(snippet))


def main() -> int:
    parser = argparse.ArgumentParser(description="测试检索与重排流程")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--query", help="用户问题，不传则交互输入")
    args = parser.parse_args()

    query = args.query.strip() if args.query else ""
    if not query:
        query = input("请输入问题: ").strip()
    if not query:
        print("问题不能为空。", file=sys.stderr)
        return 1

    try:
        config = load_config(args.config)
        pipeline = RetrievalPipeline(config)
        recall_docs, reranked_docs = pipeline.run(query)
    except Exception as exc:
        print("检索流程失败: {0}".format(exc), file=sys.stderr)
        return 1

    _print_docs("召回结果（Dense Recall）", recall_docs)
    if config.use_reranker:
        _print_docs("重排结果（CrossEncoder Rerank）", reranked_docs)
    else:
        _print_docs("重排已关闭，返回截断召回结果", reranked_docs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
