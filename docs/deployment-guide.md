# 部署运维指南

> 本文档详细说明制造业 RAG 问答系统的部署方式、配置项、运维操作和故障排查。

## 1. 部署架构

### 1.1 生产环境推荐架构

```
                    ┌─────────────┐
                    │  Nginx 反向代理 │
                    │  (负载均衡/SSL) │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ RAG App  │ │ RAG App  │ │ RAG App  │
        │ (FastAPI)│ │ (FastAPI)│ │ (FastAPI)│
        └─────┬────┘ └─────┬────┘ └─────┬────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Milvus  │ │  Redis   │ │  vLLM    │
        │ (向量库)  │ │ (缓存)   │ │ (LLM推理) │
        └──────────┘ └──────────┘ └──────────┘
```

### 1.2 最小化部署（单机）

适合测试/演示环境，所有组件在一台机器上通过 Docker Compose 运行。

## 2. Docker Compose 部署（推荐）

### 2.1 环境准备

```bash
# 安装 Docker（Ubuntu/Debian）
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER

# 安装 NVIDIA Container Toolkit（GPU 支持）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证 GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2.2 模型准备

```bash
# 创建模型目录
mkdir -p models

# 下载模型（使用 modelscope 或 huggingface）
# 方式一：ModelScope（国内推荐）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-large-zh-v1.5', cache_dir='./models')
snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir='./models')
snapshot_download('Qwen/Qwen2.5-14B-Instruct', cache_dir='./models')
"

# 方式二：HuggingFace
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./models/bge-large-zh-v1.5
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./models/bge-reranker-v2-m3
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir ./models/Qwen2.5-14B-Instruct
```

### 2.3 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑关键配置
vim .env
```

**关键配置项说明**：

| 配置项 | 说明 | 示例值 |
|--------|------|--------|
| `EMBEDDING_MODEL_NAME` | 嵌入模型路径 | `./models/bge-large-zh-v1.5` |
| `RERANKER_MODEL_NAME` | Reranker 模型路径 | `./models/bge-reranker-v2-m3` |
| `LLM_MODEL_PATH` | LLM 模型路径 | `./models/Qwen2.5-14B-Instruct` |
| `LLM_API_BASE` | vLLM 服务地址 | `http://vllm:8001/v1` |
| `MILVUS_HOST` | Milvus 地址 | `milvus` |
| `REDIS_HOST` | Redis 地址 | `redis` |
| `EMBEDDING_DEVICE` | 嵌入计算设备 | `cuda` / `cpu` |
| `CLOUD_LLM_ENABLED` | 是否启用云端兜底 | `true` |

### 2.4 启动服务

```bash
# 进入部署目录
cd deploy/docker

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f rag-app

# 查看 vLLM 日志（确认模型加载完成）
docker-compose logs -f vllm
```

### 2.5 验证部署

```bash
# 健康检查
curl http://localhost:8000/api/health

# 系统统计
curl http://localhost:8000/api/stats

# API 文档
open http://localhost:8000/docs
```

## 3. 裸机部署（无 Docker）

### 3.1 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt

# 安装系统依赖（Ubuntu）
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 poppler-utils
```

### 3.2 启动 Milvus

```bash
# 下载 Milvus  standalone 配置
wget https://github.com/milvus-io/milvus/releases/download/v2.4.4/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动 Milvus
docker-compose up -d

# 验证
docker-compose ps
```

### 3.3 启动 Redis

```bash
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine
```

### 3.4 启动 vLLM

```bash
# 启动 vLLM OpenAI 兼容服务
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-14B-Instruct \
  --served-model-name Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

### 3.5 启动 RAG 应用

```bash
# 批量入库文档
python scripts/ingest_docs.py --dir ./data/raw

# 启动 API 服务
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

## 4. 运维操作

### 4.1 文档管理

```bash
# 批量入库
python scripts/ingest_docs.py --dir /path/to/docs --recursive

# 查看向量库统计
curl http://localhost:8000/api/stats

# 删除文档（按来源）
curl -X DELETE "http://localhost:8000/api/documents?source=/path/to/doc.pdf"
```

### 4.2 数据同步

```bash
# 手动触发 OA 同步
python -c "
import asyncio
from src.integrations import DataSyncManager, OAClient
async def main():
    oa = OAClient()
    sync = DataSyncManager(oa, ingest_callback=lambda x, y: None)
    result = await sync.sync_once()
    print(result)
asyncio.run(main())
"

# 查看同步状态
cat data/sync_state.json
```

### 4.3 性能监控

```bash
# Prometheus 指标
curl http://localhost:9090/metrics

# 系统统计
curl http://localhost:8000/api/stats

# 健康检查
curl http://localhost:8000/api/health
```

### 4.4 日志管理

```bash
# 查看应用日志
docker-compose logs -f rag-app

# 查看 vLLM 日志
docker-compose logs -f vllm

# 查看 Milvus 日志
docker-compose logs -f milvus

# 日志轮转配置（loguru）
# 在 .env 中设置 LOG_LEVEL=INFO/DEBUG/WARNING
```

### 4.5 备份与恢复

```bash
# 备份 Milvus 数据
docker exec manufacturing-rag-milvus milvus backup create -n rag_backup

# 备份 Redis
docker exec manufacturing-rag-redis redis-cli BGSAVE
docker cp manufacturing-rag-redis:/data/dump.rdb ./backup/

# 恢复 Milvus
docker exec manufacturing-rag-milvus milvus backup restore -n rag_backup

# 恢复 Redis
docker cp ./backup/dump.rdb manufacturing-rag-redis:/data/dump.rdb
docker restart manufacturing-rag-redis
```

## 5. 性能调优

### 5.1 vLLM 调优

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--gpu-memory-utilization` | GPU 显存利用率 | 0.85-0.9 |
| `--max-model-len` | 最大上下文长度 | 8192 |
| `--tensor-parallel-size` | 张量并行 GPU 数 | 1（单卡）/ 2（双卡） |
| `--max-num-seqs` | 最大并发序列数 | 256 |
| `--enforce-eager` | 禁用 CUDA Graph（调试用） | false |

### 5.2 Milvus 调优

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `index_type` | 索引类型 | HNSW |
| `M` | HNSW 连接数 | 16-32 |
| `efConstruction` | 构建时搜索深度 | 200-400 |
| `ef` | 搜索时搜索深度 | 128-256 |
| `nlist` | IVF 聚类数 | sqrt(向量数) |

### 5.3 检索参数调优

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `RETRIEVAL_TOP_K` | 检索返回数量 | 10-20 |
| `RERANKER_TOP_N` | Rerank 返回数量 | 3-5 |
| `RETRIEVAL_SCORE_THRESHOLD` | 相似度阈值 | 0.5-0.7 |
| `DOCUMENT_CHUNK_SIZE` | 分块大小 | 384-512 |
| `DOCUMENT_CHUNK_OVERLAP` | 分块重叠 | 50-128 |

### 5.4 系统资源监控

```bash
# GPU 监控
watch -n 1 nvidia-smi

# 容器资源监控
docker stats

# Milvus 性能监控
docker exec manufacturing-rag-milvus milvus metrics

# Redis 性能监控
docker exec manufacturing-rag-redis redis-cli INFO stats
```

## 6. 故障排查

### 6.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 服务启动失败 | 模型路径错误 | 检查 `.env` 中模型路径是否正确 |
| GPU 内存不足 | 模型太大或并发太高 | 降低 `--gpu-memory-utilization` 或使用更小模型 |
| 检索结果为空 | 向量库为空或阈值过高 | 先执行文档入库，降低 `RETRIEVAL_SCORE_THRESHOLD` |
| 回答不相关 | 检索质量差或 Prompt 问题 | 检查分块策略，优化 Prompt，启用 Reranker |
| 响应超时 | LLM 推理慢或网络问题 | 检查 vLLM 状态，增加超时时间，启用云端兜底 |
| Milvus 连接失败 | Milvus 未启动或网络问题 | 检查 Milvus 容器状态，确认端口映射 |
| OCR 识别乱码 | 图片质量差或语言设置错误 | 检查图片清晰度，确认 `OCR_LANG=ch` |

### 6.2 日志排查

```bash
# 查看最近 100 行应用日志
docker-compose logs --tail 100 rag-app

# 查看错误日志
docker-compose logs rag-app | grep -i error

# 进入容器调试
docker exec -it manufacturing-rag-app bash

# 测试 LLM 连接
curl http://vllm:8001/v1/models

# 测试 Milvus 连接
python -c "from pymilvus import connections; connections.connect(host='milvus', port='19530'); print('OK')"
```

### 6.3 性能问题排查

```bash
# 分析慢查询
# 1. 开启 DEBUG 日志
# 在 .env 中设置 LOG_LEVEL=DEBUG

# 2. 查看各阶段耗时
# API 响应中包含 latency_ms, retrieval_latency_ms, llm_latency_ms

# 3. 检查 GPU 利用率
nvidia-smi

# 4. 检查 Milvus 检索耗时
# 在 Milvus 日志中搜索 "search" 相关日志
```

## 7. 安全配置

### 7.1 网络安全

```yaml
# docker-compose.yml 中限制端口暴露
# 只暴露 8000 端口，Milvus/Redis 不对外暴露
ports:
  - "127.0.0.1:8000:8000"  # 只监听本地
```

### 7.2 认证配置

```bash
# 在 .env 中启用认证
AUTH_ENABLED=true
JWT_SECRET_KEY=your-strong-secret-key

# 配置 SSO 对接
OA_API_BASE=http://oa.company.com/api
OA_API_TOKEN=your-token
```

### 7.3 数据加密

```bash
# 启用 HTTPS（通过 Nginx 反向代理）
# Nginx 配置 SSL 证书
# 应用层通过 JWT 认证
```

## 8. 升级指南

### 8.1 版本升级

```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
cd deploy/docker
docker-compose build --no-cache rag-app

# 滚动更新
docker-compose up -d rag-app

# 验证
curl http://localhost:8000/api/health
```

### 8.2 模型升级

```bash
# 下载新模型到 models/ 目录
# 修改 .env 中的模型路径
LLM_MODEL_PATH=./models/New-Model

# 重启 vLLM
docker-compose restart vllm

# 验证模型加载
docker-compose logs vllm | grep "Application startup complete"
```

---

**相关文档**：
- [架构设计详解](architecture.md)
- [技术选型决策](tech-selection.md)
- [效果优化历程](optimization-journey.md)
