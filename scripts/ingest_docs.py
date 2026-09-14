"""
文档批量入库脚本
用法：python scripts/ingest_docs.py --dir /path/to/docs
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.document_processor import DocumentLoader, SemanticChunker
from src.retriever import VectorStoreManager


def main():
    parser = argparse.ArgumentParser(description="批量文档入库脚本")
    parser.add_argument("--dir", type=str, required=True, help="文档目录路径")
    parser.add_argument("--recursive", action="store_true", default=True, help="是否递归子目录")
    parser.add_argument("--batch-size", type=int, default=32, help="嵌入批处理大小")
    args = parser.parse_args()

    doc_dir = Path(args.dir)
    if not doc_dir.exists():
        logger.error(f"目录不存在: {doc_dir}")
        sys.exit(1)

    logger.info(f"开始批量入库: {doc_dir}")

    # 初始化组件
    logger.info("加载嵌入模型...")
    embedding_model = SentenceTransformer(
        settings.embedding_model_name,
        device=settings.embedding_device,
    )

    loader = DocumentLoader()
    chunker = SemanticChunker()
    vector_store = VectorStoreManager()

    # 加载文档
    logger.info("加载文档...")
    documents = loader.load_directory(str(doc_dir), recursive=args.recursive)
    logger.info(f"共加载 {len(documents)} 个文档片段")

    if not documents:
        logger.warning("没有找到可处理的文档")
        return

    # 分块
    logger.info("文档分块...")
    chunks = chunker.split_documents(documents)
    logger.info(f"分块完成: {len(chunks)} 个块")

    # 生成向量
    logger.info("生成向量...")
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    # 批量入库
    logger.info("写入向量库...")
    inserted = vector_store.insert_documents(chunks, embeddings)

    logger.info(f"入库完成! 共 {inserted} 条向量")
    logger.info(f"向量库统计: {vector_store.get_stats()}")

    vector_store.close()


if __name__ == "__main__":
    main()
