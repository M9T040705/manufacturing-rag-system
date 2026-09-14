"""
FastAPI 主应用
提供问答、文档管理、系统监控等 API
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentIngestResponse,
    HealthResponse,
    SourceDocument,
    StatsResponse,
)
from ..document_processor import DocumentLoader, SemanticChunker
from ..llm import LLMClient, QA_PROMPT, SYSTEM_PROMPT
from ..retriever import HybridRetriever, Reranker, VectorStoreManager

# 全局组件
components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化组件，关闭时释放资源"""
    logger.info("正在初始化系统组件...")

    # 初始化嵌入模型
    embedding_model = SentenceTransformer(
        settings.embedding_model_name,
        device=settings.embedding_device,
    )
    components["embedding_model"] = embedding_model
    logger.info("嵌入模型加载完成")

    # 初始化向量库
    vector_store = VectorStoreManager()
    components["vector_store"] = vector_store

    # 初始化混合检索器
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_fn=lambda text: embedding_model.encode(text, normalize_embeddings=True).tolist(),
    )
    components["hybrid_retriever"] = hybrid_retriever

    # 初始化 Reranker
    if settings.reranker_top_n > 0:
        reranker = Reranker()
        components["reranker"] = reranker

    # 初始化 LLM 客户端
    llm_client = LLMClient()
    components["llm_client"] = llm_client

    # 初始化文档处理器
    components["loader"] = DocumentLoader()
    components["chunker"] = SemanticChunker()

    components["start_time"] = time.time()
    logger.info("系统初始化完成，服务已启动")

    yield

    # 关闭时释放资源
    logger.info("正在关闭系统...")
    if "llm_client" in components:
        await components["llm_client"].close()
    if "vector_store" in components:
        components["vector_store"].close()
    logger.info("系统已关闭")


app = FastAPI(
    title="制造业企业知识库 RAG 问答系统",
    description="私有化部署的制造业智能问答系统，支持文档解析、混合检索、重排序、流式输出",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 核心问答接口 =====

@app.post("/api/chat", response_model=ChatResponse, summary="智能问答")
async def chat(request: ChatRequest):
    """
    基于知识库的智能问答

    - 支持多轮对话（通过 session_id）
    - 混合检索 + 重排序
    - 返回引用来源
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # 1. 检索
        retrieval_start = time.time()
        retriever: HybridRetriever = components["hybrid_retriever"]
        retrieved_docs = retriever.retrieve(request.question, top_k=request.top_k)
        retrieval_latency = int((time.time() - retrieval_start) * 1000)

        # 2. 重排序
        if request.use_rerank and "reranker" in components and retrieved_docs:
            reranker: Reranker = components["reranker"]
            retrieved_docs = reranker.rerank(request.question, retrieved_docs)

        # 3. 构建上下文
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("filename", "未知")
            page = doc.metadata.get("page", "")
            heading = doc.metadata.get("heading", "")
            context_parts.append(
                f"【文档{i}】来源: {source}, 页码: {page}, 标题: {heading}\n{doc.page_content}"
            )
        context = "\n---\n".join(context_parts) if context_parts else "无相关参考资料"

        # 4. 调用 LLM
        llm_start = time.time()
        llm_client: LLMClient = components["llm_client"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QA_PROMPT.format(context=context, question=request.question)},
        ]
        answer = await llm_client.chat(messages)
        llm_latency = int((time.time() - llm_start) * 1000)

        # 5. 构建来源列表
        sources = [
            SourceDocument(
                content=doc.page_content[:500],
                source=doc.metadata.get("filename", "未知"),
                page=doc.metadata.get("page"),
                score=doc.metadata.get("rerank_score", doc.metadata.get("score", 0)),
                heading=doc.metadata.get("heading"),
            )
            for doc in retrieved_docs[:5]
        ]

        total_latency = int((time.time() - start_time) * 1000)
        logger.info(f"问答完成: session={session_id}, 耗时={total_latency}ms, 来源数={len(sources)}")

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            sources=sources,
            latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            llm_latency_ms=llm_latency,
        )

    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@app.post("/api/chat/stream", summary="流式问答")
async def chat_stream(request: ChatRequest):
    """流式输出问答结果（SSE）"""
    session_id = request.session_id or str(uuid.uuid4())

    async def generate():
        try:
            # 检索
            retriever: HybridRetriever = components["hybrid_retriever"]
            retrieved_docs = retriever.retrieve(request.question)

            if request.use_rerank and "reranker" in components and retrieved_docs:
                reranker: Reranker = components["reranker"]
                retrieved_docs = reranker.rerank(request.question, retrieved_docs)

            # 构建上下文
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"【文档{i}】{doc.page_content}")
            context = "\n---\n".join(context_parts)

            # 流式生成
            llm_client: LLMClient = components["llm_client"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": QA_PROMPT.format(context=context, question=request.question)},
            ]

            async for chunk in llm_client.chat_stream(messages):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"流式问答失败: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ===== 文档管理接口 =====

@app.post("/api/documents/ingest", response_model=DocumentIngestResponse, summary="文档入库")
async def ingest_document(file: UploadFile = File(...)):
    """上传文档并解析入库到向量库"""
    try:
        # 保存临时文件
        import tempfile
        import os

        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 加载文档
        loader: DocumentLoader = components["loader"]
        documents = loader.load(tmp_path)

        # 分块
        chunker: SemanticChunker = components["chunker"]
        chunks = chunker.split_documents(documents)

        # 生成向量并入库
        embedding_model = components["embedding_model"]
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

        vector_store: VectorStoreManager = components["vector_store"]
        inserted = vector_store.insert_documents(chunks, embeddings)

        # 清理临时文件
        os.unlink(tmp_path)

        return DocumentIngestResponse(
            success=True,
            filename=file.filename,
            chunks_count=len(chunks),
            vectors_count=inserted,
            message=f"文档入库成功，共 {inserted} 个向量",
        )

    except Exception as e:
        logger.error(f"文档入库失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档入库失败: {str(e)}")


@app.delete("/api/documents", summary="删除文档")
async def delete_document(source: str):
    """根据来源路径删除文档"""
    try:
        vector_store: VectorStoreManager = components["vector_store"]
        deleted = vector_store.delete_by_source(source)
        return {"success": True, "deleted_count": deleted}
    except Exception as e:
        logger.error(f"文档删除失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 系统监控接口 =====

@app.get("/api/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """检查各组件健康状态"""
    # Milvus
    milvus_status = "healthy"
    try:
        vector_store: VectorStoreManager = components["vector_store"]
        vector_store.get_stats()
    except Exception:
        milvus_status = "unhealthy"

    # LLM
    llm_status = "healthy"
    try:
        llm_client: LLMClient = components["llm_client"]
        # 简单的 ping 测试可以在这里加
    except Exception:
        llm_status = "unhealthy"

    return HealthResponse(
        status="healthy" if milvus_status == "healthy" else "degraded",
        milvus=milvus_status,
        llm=llm_status,
        redis="unknown",
    )


@app.get("/api/stats", response_model=StatsResponse, summary="系统统计")
async def get_stats():
    """获取系统统计信息"""
    vector_store: VectorStoreManager = components["vector_store"]
    stats = vector_store.get_stats()

    uptime = time.time() - components.get("start_time", time.time())

    return StatsResponse(
        total_documents=stats["num_entities"],
        total_chunks=stats["num_entities"],
        collection_name=stats["collection_name"],
        model_name=settings.llm_model_name,
        embedding_model=settings.embedding_model_name,
        uptime_seconds=round(uptime, 2),
    )


@app.get("/", summary="根路径")
async def root():
    return {
        "name": "制造业企业知识库 RAG 问答系统",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
