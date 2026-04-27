from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


def _normalize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("messages 中的每条消息必须是字典")
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not role:
            raise ValueError("消息缺少 role 字段")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("messages 不能为空")
    return normalized


class BaseLLMClient(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate(self, messages):  # type: (List[Dict[str, str]]) -> str
        """根据消息列表生成回答文本。"""
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """本地测试专用 Mock 客户端。"""

    def generate(self, messages):  # type: (List[Dict[str, str]]) -> str
        normalized = _normalize_messages(messages)
        user_messages = [m["content"] for m in normalized if m["role"] == "user"]
        last_user = user_messages[-1] if user_messages else ""
        return "【Mock回答】已收到请求。以下为测试输出，不代表真实医疗建议。\n{0}".format(
            last_user[:200]
        )


class TongyiLLMClient(BaseLLMClient):
    """阿里云百炼 Tongyi 客户端。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._chat_model = None

    def _get_chat_model(self):
        if self._chat_model is not None:
            return self._chat_model

        try:
            from dotenv import load_dotenv
            from langchain_community.chat_models.tongyi import ChatTongyi
        except ImportError:
            raise ImportError(
                "缺少 Tongyi 依赖，请安装 python-dotenv 与 langchain-community。"
            )

        load_dotenv()
        self._chat_model = ChatTongyi(model=self.config.llm_model)
        return self._chat_model

    def generate(self, messages):  # type: (List[Dict[str, str]]) -> str
        normalized = _normalize_messages(messages)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            raise ImportError("缺少 langchain-core，无法构建 Tongyi 消息格式。")

        lc_messages = []
        for msg in normalized:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Tongyi 在本项目中主要使用 system + user，这里把 assistant 历史并入 human。
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))

        response = self._get_chat_model().invoke(lc_messages)
        return str(getattr(response, "content", response))


class LocalChatGLMClient(BaseLLMClient):
    """本地 ChatGLM 客户端，模型延迟加载。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None

    def _ensure_model_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "缺少本地模型依赖，请安装 torch 与 transformers 后重试。"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            self.config.llm_model,
            trust_remote_code=True,
        )

        device = str(self.config.generation_device)
        if device.startswith("cuda") and hasattr(model, "half"):
            model = model.half()
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model = model.eval()

        self._tokenizer = tokenizer
        self._model = model

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append("【系统】\n{0}".format(content))
            elif role == "assistant":
                parts.append("【助手】\n{0}".format(content))
            else:
                parts.append("【用户】\n{0}".format(content))
        return "\n\n".join(parts)

    def generate(self, messages):  # type: (List[Dict[str, str]]) -> str
        normalized = _normalize_messages(messages)
        self._ensure_model_loaded()

        prompt = self._build_prompt(normalized)
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("本地模型未正确初始化。")

        if hasattr(model, "chat"):
            response = model.chat(tokenizer, prompt, history=[])
            if isinstance(response, tuple):
                return str(response[0])
            return str(response)

        try:
            import torch
        except ImportError:
            raise ImportError("缺少 torch，无法调用本地模型 generate。")

        device = str(self.config.generation_device)
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def create_llm_client(config: Any) -> BaseLLMClient:
    provider = str(getattr(config, "llm_provider", "")).strip().lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "tongyi":
        return TongyiLLMClient(config)
    if provider == "local":
        return LocalChatGLMClient(config)
    raise ValueError("不支持的 llm_provider: {0}".format(provider))
