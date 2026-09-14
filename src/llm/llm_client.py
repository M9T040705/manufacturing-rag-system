"""
LLM 客户端
支持本地 vLLM 部署 + 云端 DeepSeek 兜底
统一 OpenAI 兼容接口，支持流式输出
"""
from typing import AsyncGenerator, List, Optional

import httpx
from loguru import logger

from config.settings import settings


class LLMClient:
    """统一 LLM 调用客户端，支持本地 + 云端兜底"""

    def __init__(self):
        self.local_client = httpx.AsyncClient(
            base_url=settings.llm_api_base,
            timeout=120.0,
        )
        self.local_headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

        if settings.cloud_llm_enabled and settings.cloud_llm_api_key:
            self.cloud_client = httpx.AsyncClient(
                base_url=settings.cloud_llm_api_base,
                timeout=60.0,
            )
            self.cloud_headers = {"Authorization": f"Bearer {settings.cloud_llm_api_key}"}
        else:
            self.cloud_client = None
            self.cloud_headers = None

        logger.info(
            f"LLM 客户端初始化: 本地模型={settings.llm_model_name}, "
            f"云端兜底={'启用' if self.cloud_client else '禁用'}"
        )

    async def chat(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cloud_fallback: bool = True,
    ) -> str:
        """
        非流式对话

        Args:
            messages: 消息列表 [{"role": "user"/"system", "content": "..."}]
            temperature: 温度
            max_tokens: 最大生成 token
            use_cloud_fallback: 本地失败时是否使用云端兜底

        Returns:
            生成的文本
        """
        try:
            return await self._chat_local(messages, temperature, max_tokens)
        except Exception as e:
            logger.warning(f"本地 LLM 调用失败: {e}")
            if use_cloud_fallback and self.cloud_client:
                logger.info("切换到云端 LLM 兜底")
                return await self._chat_cloud(messages, temperature, max_tokens)
            raise

    async def chat_stream(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式对话（SSE）

        Yields:
            逐段生成的文本
        """
        payload = {
            "model": settings.llm_model_name,
            "messages": messages,
            "temperature": temperature or settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": True,
        }

        try:
            async with self.local_client.stream(
                "POST", "/chat/completions", json=payload, headers=self.local_headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as e:
            logger.error(f"流式调用失败: {e}")
            raise

    async def _chat_local(
        self,
        messages: List[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """调用本地 vLLM"""
        payload = {
            "model": settings.llm_model_name,
            "messages": messages,
            "temperature": temperature or settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "top_p": settings.llm_top_p,
        }

        response = await self.local_client.post(
            "/chat/completions", json=payload, headers=self.local_headers
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def _chat_cloud(
        self,
        messages: List[dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """调用云端 DeepSeek"""
        if not self.cloud_client:
            raise RuntimeError("云端 LLM 未配置")

        payload = {
            "model": settings.cloud_llm_model,
            "messages": messages,
            "temperature": temperature or 0.7,
            "max_tokens": max_tokens or 2048,
        }

        response = await self.cloud_client.post(
            "/chat/completions", json=payload, headers=self.cloud_headers
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def close(self):
        """关闭客户端连接"""
        await self.local_client.aclose()
        if self.cloud_client:
            await self.cloud_client.aclose()
