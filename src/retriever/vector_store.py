"""
Milvus 向量库管理
负责集合创建、文档插入、向量检索、索引管理
"""
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config.settings import settings


class VectorStoreManager:
    """Milvus 向量库管理器"""

    def __init__(self):
        self.collection_name = settings.milvus_collection_name
        self.dim = settings.milvus_dim
        self.collection: Optional[Collection] = None
        self._connect()
        self._init_collection()

    def _connect(self):
        """连接 Milvus"""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
            )
            logger.info(f"Milvus 连接成功: {settings.milvus_host}:{settings.milvus_port}")
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise

    def _init_collection(self):
        """初始化集合（不存在则创建）"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"集合已存在: {self.collection_name}, 实体数: {self.collection.num_entities}")
            return

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields=fields, description="制造业知识库向量集合")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # 创建索引
        index_params = {
            "index_type": settings.milvus_index_type,
            "metric_type": settings.milvus_metric_type,
            "params": {"M": 16, "efConstruction": 200},
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        self.collection.load()
        logger.info(f"集合创建成功: {self.collection_name}")

    def insert_documents(self, documents: List[Document], embeddings: List[List[float]]) -> int:
        """
        批量插入文档和向量

        Args:
            documents: 文档列表
            embeddings: 对应的向量列表

        Returns:
            插入数量
        """
        if len(documents) != len(embeddings):
            raise ValueError("文档数量与向量数量不匹配")

        if not documents:
            return 0

        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        data = [ids, embeddings, texts, metadatas]

        try:
            self.collection.insert(data)
            self.collection.flush()
            logger.info(f"成功插入 {len(documents)} 条文档")
            return len(documents)
        except Exception as e:
            logger.error(f"文档插入失败: {e}")
            raise

    def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        向量相似度检索

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            检索到的文档列表
        """
        top_k = top_k or settings.retrieval_top_k
        score_threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

        search_params = {"metric_type": settings.milvus_metric_type, "params": {"ef": 128}}

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata"],
        )

        documents = []
        for hits in results:
            for hit in hits:
                # Cosine 相似度转换为 0-1 分数
                score = hit.score
                if settings.milvus_metric_type == "COSINE":
                    score = (score + 1) / 2  # Milvus 返回的是 -1~1

                if score < score_threshold:
                    continue

                entity = hit.entity
                doc = Document(
                    page_content=entity.get("text", ""),
                    metadata={
                        **entity.get("metadata", {}),
                        "score": score,
                        "retrieval_method": "vector",
                    }
                )
                documents.append(doc)

        logger.debug(f"向量检索返回 {len(documents)} 条结果")
        return documents

    def delete_by_source(self, source: str) -> int:
        """根据来源文件删除所有相关文档"""
        expr = f'metadata["source"] == "{source}"'
        try:
            results = self.collection.delete(expr)
            self.collection.flush()
            deleted = results.delete_count if hasattr(results, "delete_count") else 0
            logger.info(f"删除来源 {source} 的文档，删除数量: {deleted}")
            return deleted
        except Exception as e:
            logger.error(f"删除失败: {e}")
            raise

    def get_stats(self) -> dict:
        """获取集合统计信息"""
        return {
            "collection_name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "dim": self.dim,
            "index_type": settings.milvus_index_type,
        }

    def close(self):
        """关闭连接"""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")
        logger.info("Milvus 连接已关闭")
