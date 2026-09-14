"""LLM 推理模块：本地 vLLM + 云端兜底"""
from .llm_client import LLMClient
from .prompt_templates import QA_PROMPT, QUERY_REWRITE_PROMPT, SUMMARY_PROMPT

__all__ = ["LLMClient", "QA_PROMPT", "QUERY_REWRITE_PROMPT", "SUMMARY_PROMPT"]
