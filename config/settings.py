"""
全局配置管理
使用 pydantic-settings 从环境变量和 .env 文件加载配置
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用基础配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 应用
    app_name: str = "manufacturing-rag-system"
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # 嵌入模型
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # Rerank
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    reranker_top_n: int = 5

    # LLM
    llm_model_path: str = "/models/Qwen2.5-14B-Instruct"
    llm_api_base: str = "http://localhost:8001/v1"
    llm_api_key: str = "EMPTY"
    llm_model_name: str = "Qwen2.5-14B-Instruct"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.3
    llm_top_p: float = 0.9

    # 云端兜底
    cloud_llm_enabled: bool = True
    cloud_llm_api_base: str = "https://api.deepseek.com/v1"
    cloud_llm_api_key: str = ""
    cloud_llm_model: str = "deepseek-chat"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "manufacturing_knowledge"
    milvus_dim: int = 1024
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "COSINE"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_cache_ttl: int = 3600

    # 文档处理
    document_chunk_size: int = 512
    document_chunk_overlap: int = 64
    document_max_file_size: str = "50MB"
    document_supported_formats: str = ".pdf,.docx,.doc,.pptx,.txt,.png,.jpg,.jpeg"
    ocr_enabled: bool = True
    ocr_lang: str = "ch"
    ocr_use_gpu: bool = True

    # 检索
    retrieval_top_k: int = 10
    retrieval_score_threshold: float = 0.5
    hybrid_retrieval_weight_vector: float = 0.7
    hybrid_retrieval_weight_bm25: float = 0.3

    # OA 对接
    oa_api_base: str = "http://oa.company.com/api"
    oa_api_token: str = ""
    oa_sync_enabled: bool = True
    oa_sync_cron: str = "0 2 * * *"
    oa_sync_batch_size: int = 100

    # 安全
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440
    auth_enabled: bool = True

    # 监控
    prometheus_enabled: bool = True
    prometheus_port: int = 9090


@lru_cache()
def get_settings() -> AppSettings:
    """获取全局配置单例"""
    return AppSettings()


settings = get_settings()
