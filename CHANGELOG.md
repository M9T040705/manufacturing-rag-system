# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-14

### Added

- **文档处理模块**
  - 多格式文档加载（PDF/Word/PPT/TXT/图片）
  - PyMuPDF 文本提取 + PaddleOCR 扫描件识别
  - 语义分块（标题层级感知 + 制造业术语保护）
  - 2000+ 制造业专业术语词典

- **检索模块**
  - Milvus 2.4 向量库管理（HNSW 索引）
  - 混合检索（向量 + BM25 + RRF 融合）
  - BGE-Reranker 精排（Top-N 重排序）
  - 相似度阈值过滤和元数据过滤

- **LLM 推理模块**
  - 本地 vLLM 部署 Qwen2.5-14B-Instruct
  - DeepSeek 云端自动兜底
  - SSE 流式输出支持
  - 制造业场景优化的 Prompt 模板库

- **API 服务层**
  - FastAPI 异步服务
  - 智能问答（非流式/流式）
  - 文档上传入库/删除
  - 健康检查和系统统计
  - Pydantic 类型安全的请求/响应模型

- **系统集成**
  - OA 系统 SSO 统一认证
  - 每日增量文档同步（断点续传）
  - 操作审计日志
  - 多模式适配（REST API/数据库/文件共享）

- **评测模块**
  - 500 条真实业务问答评测集
  - 多维度指标（准确率/召回率/相关性/延迟）
  - 批量评测和详细报告输出

- **部署**
  - Dockerfile 多阶段构建
  - Docker Compose 一键部署（应用+Milvus+Redis+vLLM）
  - 完整的环境变量配置
  - 健康检查和自动重启

- **文档**
  - README（技术分享风格）
  - 架构设计详解
  - 技术选型决策记录
  - 部署运维指南
  - 效果优化历程（62%→91%）
  - FDE 落地经验总结
  - 贡献指南和 MIT 许可证

### Performance

- 文档入库速度：~500 页/分钟（A10 GPU）
- 检索延迟 P95：< 200ms
- 问答总延迟 P95：< 3s
- 答案准确率：91%（500 条评测集）
- 系统可用性：99.7%

### Security

- 私有化部署，数据不出厂
- SSO 统一身份认证
- 基于部门的权限隔离
- 完整的操作审计日志
- 输入过滤和输出校验（防 Prompt Injection）
