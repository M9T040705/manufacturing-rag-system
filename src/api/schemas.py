"""
API 请求/响应模型定义
"""
from typing import List, Optional

from pydantic import BaseModel, Field


# ===== 请求模型 =====

class ChatRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    stream: bool = Field(False, description="是否流式输出")
    top_k: Optional[int] = Field(None, description="检索返回数量")
    use_rerank: bool = Field(True, description="是否使用重排序")


class DocumentUploadRequest(BaseModel):
    """文档上传请求（元数据）"""
    filename: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    source: Optional[str] = Field(None, description="来源系统")
    department: Optional[str] = Field(None, description="所属部门")
    tags: Optional[List[str]] = Field(None, description="标签")


class DocumentDeleteRequest(BaseModel):
    """文档删除请求"""
    source: str = Field(..., description="文档来源路径")


class EvaluationRequest(BaseModel):
    """评测请求"""
    eval_set_path: str = Field(..., description="评测集路径")
    metrics: List[str] = Field(default=["accuracy", "relevance", "latency"])


# ===== 响应模型 =====

class SourceDocument(BaseModel):
    """引用来源文档"""
    content: str = Field(..., description="文档内容片段")
    source: str = Field(..., description="来源文件")
    page: Optional[int] = Field(None, description="页码")
    score: float = Field(..., description="相似度分数")
    heading: Optional[str] = Field(None, description="所属标题")


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str = Field(..., description="回答内容")
    session_id: str = Field(..., description="会话ID")
    sources: List[SourceDocument] = Field(default_factory=list, description="引用来源")
    latency_ms: int = Field(..., description="总耗时(毫秒)")
    retrieval_latency_ms: int = Field(..., description="检索耗时(毫秒)")
    llm_latency_ms: int = Field(..., description="LLM耗时(毫秒)")


class DocumentIngestResponse(BaseModel):
    """文档入库响应"""
    success: bool = Field(..., description="是否成功")
    filename: str = Field(..., description="文件名")
    chunks_count: int = Field(..., description="分块数量")
    vectors_count: int = Field(..., description="入库向量数量")
    message: str = Field(..., description="处理消息")


class StatsResponse(BaseModel):
    """系统统计响应"""
    total_documents: int = Field(..., description="总文档数")
    total_chunks: int = Field(..., description="总分块数")
    collection_name: str = Field(..., description="向量集合名")
    model_name: str = Field(..., description="LLM模型名")
    embedding_model: str = Field(..., description="嵌入模型名")
    uptime_seconds: float = Field(..., description="运行时长(秒)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    milvus: str = Field(..., description="Milvus状态")
    llm: str = Field(..., description="LLM状态")
    redis: str = Field(..., description="Redis状态")
