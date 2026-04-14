# llm_utils.py
# Python 3.9 compatible
# 支持 XIAOHUAPI / OpenAI-compatible /v1/chat/completions
# ✅ 自动降级 fallback：当 gpt-4o 无可用渠道/模型不存在时，自动尝试其他模型
#
# ✅ 修复点：
# 1) base_url 既可传 .../v1，也可传 .../v1/chat/completions（自动归一化）
# 2) 统一拼接 chat_url，避免重复 /chat/completions 或缺失
# 3) list_models 路径推导更稳（从 /v1 推到 /v1/models）
# 4) 请求错误信息更清晰（把 status_code、text 摘要打印出来）

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class LLMError(RuntimeError):
    pass


def extract_json_obj(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取第一个 JSON 对象并解析。
    兼容 ```json ... ``` 或前后有自然语言说明的情况。
    """
    if text is None:
        raise ValueError("模型输出为空")

    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        raise ValueError("未找到 JSON 对象，原始输出:\n{}".format(text[:800]))

    json_str = m.group(0).strip()

    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError("JSON 解析失败: {}\n提取到的 JSON:\n{}".format(e, json_str[:1200]))


def _normalize_base_url(base_url: str) -> str:
    """
    允许传：
      - https://chat.xiaohuapi.site/v1
      - https://chat.xiaohuapi.site/v1/chat/completions
    统一规整成：
      - https://chat.xiaohuapi.site/v1
    """
    u = (base_url or "").strip().rstrip("/")
    if not u:
        raise ValueError("base_url 不能为空")

    # 如果传的是 .../chat/completions，就裁到 .../v1
    # 兼容 .../v1/chat/completions 或 .../chat/completions
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")].rstrip("/")

    # 如果 u 末尾不是 /v1，也不强行补（有些网关不是 /v1，但你现在是 /v1）
    return u


@dataclass
class XiaoHuAPIClient:
    """
    XIAOHUAPI（OpenAI兼容）最小客户端：
    - base_url 建议配置到 .../v1
    - POST {base_url}/chat/completions

    ✅ 自动降级机制：
    若报错包含 model_not_found / 无可用渠道 / distributor 等信息，
    则自动依次尝试 fallback_models。
    """

    api_key: str
    base_url: str = "https://chat.xiaohuapi.site/v1"
    model: str = "gpt-4o"
    timeout: int = 120
    max_retries: int = 3
    backoff_sec: float = 1.5

    # ✅ 你可以按需修改/补充这里的模型列表（按优先级从高到低）
    fallback_models: Optional[List[str]] = None

    def __post_init__(self) -> None:
        # 归一化 base_url
        self.base_url = _normalize_base_url(self.base_url)

        # chat_url 永远是 {base_url}/chat/completions
        self.chat_url = self.base_url.rstrip("/") + "/chat/completions"

        # 默认 fallback 列表
        if self.fallback_models is None:
            # 注意：中转可能不支持所有模型名，你可以按实际可用的改
            self.fallback_models = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"]

    def _should_fallback(self, err_msg: str) -> bool:
        """
        判断是否触发模型自动降级。
        """
        if not err_msg:
            return False
        keys = ["model_not_found", "无可用渠道", "distributor", "channel", "not found"]
        low = err_msg.lower()
        for k in keys:
            if k.lower() in low:
                return True
        return False

    def _post(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """
        单次请求封装：成功返回 json，失败抛异常
        """
        r = requests.post(self.chat_url, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise LLMError("HTTP {}: {}".format(r.status_code, r.text[:800]))
        try:
            return r.json()
        except Exception:
            raise LLMError("HTTP {}: 非JSON响应: {}".format(r.status_code, r.text[:800]))

    def chat_raw(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None

        # 先按正常模型重试 max_retries 次
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._post(payload, headers)
            except Exception as e:
                last_err = e
                time.sleep(self.backoff_sec * attempt)

        # 如果正常模型失败，判断是否需要 fallback
        msg = str(last_err) if last_err else ""
        if self._should_fallback(msg):
            # 逐个尝试 fallback_models
            for fm in (self.fallback_models or []):
                try:
                    payload["model"] = fm
                    data = self._post(payload, headers)
                    print("[LLM fallback] switched to model:", fm)
                    return data
                except Exception as e2:
                    last_err = e2
                    continue

        raise LLMError("LLM 请求失败（重试后仍失败）：{}".format(last_err))

    def chat_text(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        data = self.chat_raw(messages, **kwargs)
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise LLMError("返回结构不符合预期：{}".format(str(data)[:1200]))

    def chat_json(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        text = self.chat_text(messages, **kwargs)
        return extract_json_obj(text)

    def list_models(self) -> Dict[str, Any]:
        """
        如果网关支持 /models，可用于探测可用模型。
        默认：GET {base_url}/models
        """
        url = self.base_url.rstrip("/") + "/models"
        headers = {"Authorization": "Bearer {}".format(self.api_key)}
        r = requests.get(url, headers=headers, timeout=self.timeout)
        if r.status_code >= 400:
            raise LLMError("HTTP {}: {}".format(r.status_code, r.text[:800]))
        try:
            return r.json()
        except Exception:
            raise LLMError("HTTP {}: 非JSON响应: {}".format(r.status_code, r.text[:800]))
