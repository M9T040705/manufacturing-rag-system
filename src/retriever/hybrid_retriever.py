"""
混合检索器
结合向量检索 + BM25 关键词检索，加权融合后返回结果
"""
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings
from .vector_store import VectorStoreManager


class HybridRetriever:
    """向量 + BM25 混合检索器"""

    def __init__(self, vector_store: VectorStoreManager, embedding_fn):
        """
        Args:
            vector_store: 向量库管理器
            embedding_fn: 嵌入函数，输入文本返回向量
        """
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        self.bm25_corpus: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._weight_vector = settings.hybrid_retrieval_weight_vector
        self._weight_bm25 = settings.hybrid_retrieval_weight_bm25

    def build_bm25_index(self, documents: List[Document]):
        """构建 BM25 索引"""
        self.bm25_corpus = documents
        # 简单中文分词：按字符级 + 常见词
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 索引构建完成，文档数: {len(documents)}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        混合检索

        流程：
        1. 向量检索获取 top_k 候选
        2. BM25 检索获取 top_k 候选
        3. 归一化分数后加权融合
        4. 按融合分数排序返回
        """
        top_k = top_k or settings.retrieval_top_k

        # 1. 向量检索
        query_vector = self.embedding_fn(query)
        vector_results = self.vector_store.search(query_vector, top_k=top_k * 2)

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        # 3. 融合
        fused = self._fuse_results(vector_results, bm25_results)

        # 4. 排序并返回 top_k
        fused.sort(key=lambda x: x.metadata.get("fused_score", 0), reverse=True)
        results = fused[:top_k]

        logger.info(
            f"混合检索完成: 向量={len(vector_results)}, BM25={len(bm25_results)}, "
            f"融合后={len(results)}"
        )
        return results

    def _bm25_search(self, query: str, top_k: int) -> List[Document]:
        """BM25 关键词检索"""
        if not self.bm25 or not self.bm25_corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 按分数排序取 top_k
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        max_score = max(scores) if scores.max() > 0 else 1
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            doc = self.bm25_corpus[idx]
            doc.metadata["bm25_score"] = float(scores[idx] / max_score)
            doc.metadata["retrieval_method"] = "bm25"
            results.append(doc)

        return results

    def _fuse_results(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
    ) -> List[Document]:
        """
        融合两路检索结果
        使用 RRF (Reciprocal Rank Fusion) + 加权
        """
        # 用文本内容作为去重 key
        result_map = {}

        # 向量结果
        for rank, doc in enumerate(vector_results):
            key = doc.page_content[:100]  # 用前100字作为唯一标识
            vector_score = doc.metadata.get("score", 0)
            rrf_score = 1 / (rank + 60)  # RRF 公式，k=60

            if key in result_map:
                existing = result_map[key]
                existing.metadata["vector_score"] = vector_score
                existing.metadata["vector_rank"] = rank
                existing.metadata["fused_score"] = (
                    existing.metadata.get("fused_score", 0)
                    + self._weight_vector * (vector_score * 0.7 + rrf_score * 0.3)
                )
            else:
                doc.metadata["vector_score"] = vector_score
                doc.metadata["vector_rank"] = rank
                doc.metadata["fused_score"] = self._weight_vector * (vector_score * 0.7 + rrf_score * 0.3)
                result_map[key] = doc

        # BM25 结果
        for rank, doc in enumerate(bm25_results):
            key = doc.page_content[:100]
            bm25_score = doc.metadata.get("bm25_score", 0)
            rrf_score = 1 / (rank + 60)

            if key in result_map:
                existing = result_map[key]
                existing.metadata["bm25_score"] = bm25_score
                existing.metadata["bm25_rank"] = rank
                existing.metadata["fused_score"] = (
                    existing.metadata.get("fused_score", 0)
                    + self._weight_bm25 * (bm25_score * 0.7 + rrf_score * 0.3)
                )
                existing.metadata["retrieval_method"] = "hybrid"
            else:
                doc.metadata["bm25_score"] = bm25_score
                doc.metadata["bm25_rank"] = rank
                doc.metadata["fused_score"] = self._weight_bm25 * (bm25_score * 0.7 + rrf_score * 0.3)
                result_map[key] = doc

        return list(result_map.values())

    def _tokenize(self, text: str) -> List[str]:
        """简单中文分词：字符级 + 2-gram"""
        # 移除标点和空白
        import re
        clean = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        # 字符级
        chars = list(clean)
        # 2-gram
        bigrams = [clean[i:i+2] for i in range(len(clean) - 1)] if len(clean) > 1 else []
        return chars + bigrams
