import json
from typing import Dict, List


REQUIRED_FIELDS = {
    "id",
    "type",
    "question",
    "ground_truth",
    "expected_behavior",
}

ALLOWED_EXPECTED_BEHAVIORS = {"answer", "refuse", "safety_refuse"}


def load_eval_dataset(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except ValueError as exc:
                raise ValueError("第 {0} 行 JSON 解析失败: {1}".format(idx, exc))
            if not isinstance(item, dict):
                raise ValueError("第 {0} 行必须是 JSON 对象".format(idx))

            missing = REQUIRED_FIELDS - set(item.keys())
            if missing:
                raise ValueError("第 {0} 行缺少字段: {1}".format(idx, sorted(missing)))

            behavior = str(item.get("expected_behavior", "")).strip()
            if behavior not in ALLOWED_EXPECTED_BEHAVIORS:
                raise ValueError(
                    "第 {0} 行 expected_behavior 非法: {1}".format(idx, behavior)
                )

            rows.append(
                {
                    "id": str(item["id"]),
                    "type": str(item["type"]),
                    "question": str(item["question"]),
                    "ground_truth": str(item["ground_truth"]),
                    "expected_behavior": behavior,
                }
            )

    if not rows:
        raise ValueError("评测数据为空: {0}".format(path))
    return rows
