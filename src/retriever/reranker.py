"""
重排序器 (Reranker)
基于 BGE-Reranker 对检索结果进行精排，提升最终答案准确率
"""
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from config.settings import settings


class Reranker:
    """BGE Reranker 重排序器"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._initialized = True

    def _load_model(self):
        """懒加载 Reranker 模型"""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(
                    settings.reranker_model_name,
                    use_fp16=True,
                    device=settings.reranker_device,
                )
                logger.info(f"Reranker 模型加载成功: {settings.reranker_model_name}")
            except Exception as e:
                logger.error(f"Reranker 加载失败: {e}")
                raise
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_n: 返回前 N 个

        Returns:
            重排序后的文档列表
        """
        top_n = top_n or settings.reranker_top_n

        if not documents:
            return []

        if len(documents) <= top_n:
            # 数量不足，直接返回但仍计算分数
            pass

        model = self._load_model()

        # 构建 (query, doc) 对
        pairs = [[query, doc.page_content] for doc in documents]

        # 计算分数
        scores = model.compute_score(pairs, normalize=True)

        # 确保 scores 是列表
        if not isinstance(scores, list):
            scores = [scores]

        # 附加分数并排序
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        documents.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)

        results = documents[:top_n]
        logger.info(
            f"重排序完成: 输入 {len(documents)} 条, 返回 {len(results)} 条, "
            f"最高分: {results[0].metadata.get('rerank_score', 0):.4f}"
        )
        return results
