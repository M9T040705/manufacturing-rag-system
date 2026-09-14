# 贡献指南

感谢你对 Manufacturing RAG System 的关注！我们欢迎任何形式的贡献，包括但不限于：

- 🐛 提交 Bug 报告
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复或新功能
- ❓ 回答 Issue 中的问题

## 开发环境搭建

### 1. Fork 并克隆项目

```bash
# Fork 项目到你的 GitHub 账号
git clone https://github.com/your-username/manufacturing-rag-system.git
cd manufacturing-rag-system
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
pip install pytest black isort  # 开发工具
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，配置你的本地开发环境
```

### 5. 启动依赖服务

```bash
cd deploy/docker
docker-compose up -d milvus etcd minio redis
```

### 6. 运行测试

```bash
pytest tests/ -v
```

## 代码规范

### Python 代码风格

我们遵循 [PEP 8](https://peps.python.org/pep-0008/) 规范，并使用以下工具保证代码质量：

```bash
# 代码格式化
black src/ tests/

# 导入排序
isort src/ tests/

# 类型检查（可选）
mypy src/
```

### 提交信息规范

我们采用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型**：

| Type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 Bug |
| `docs` | 文档更新 |
| `style` | 代码格式（不影响功能） |
| `refactor` | 重构（既不修复 Bug 也不添加功能） |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `chore` | 构建/工具/依赖相关 |
| `ci` | CI/CD 相关 |

**示例**：

```
feat(retriever): add hybrid search with BM25

- Implement BM25 keyword search
- Add RRF fusion algorithm
- Improve professional term query accuracy by 23%

Closes #123
```

## 提交流程

### 1. 创建功能分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

### 2. 编写代码和测试

- 新功能必须包含单元测试
- Bug 修复必须包含回归测试
- 确保所有测试通过

### 3. 提交代码

```bash
git add .
git commit -m "feat: your commit message"
```

### 4. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request，描述你的改动。

### 5. Code Review

- 维护者会在 3 个工作日内回复
- 根据 Review 意见修改代码
- 至少 1 个维护者 Approve 后才能合并

## Pull Request 要求

一个合格的 PR 应该：

1. ✅ 标题清晰，符合 Conventional Commits 规范
2. ✅ 描述清楚改动的目的和实现方式
3. ✅ 包含相关的测试用例
4. ✅ 所有测试通过
5. ✅ 代码通过 black 和 isort 格式化
6. ✅ 不包含无关的改动（一个 PR 只做一件事）
7. ✅ 更新相关文档（如果需要）

## Issue 指南

### Bug 报告

提交 Bug 时请包含以下信息：

- **环境信息**：OS、Python 版本、GPU 型号、驱动版本
- **复现步骤**：详细的操作步骤
- **预期行为**：你期望发生什么
- **实际行为**：实际发生了什么
- **错误日志**：完整的错误堆栈
- **复现代码**：最小可复现代码片段

### 功能建议

提交功能建议时请说明：

- **问题场景**：你遇到了什么问题
- **期望方案**：你希望怎么解决
- **替代方案**：你考虑过的其他方案
- **额外上下文**：任何相关的背景信息

## 文档贡献

文档改进同样非常欢迎！你可以：

- 修复文档中的错别字或错误
- 补充缺失的说明或示例
- 改进文档结构和可读性
- 添加新的教程或最佳实践

文档位于 `docs/` 目录和 `README.md`。

## 社区行为准则

我们致力于营造一个开放、友好、包容的社区环境。所有参与者都应遵守以下准则：

- 🤝 尊重不同的观点和经验
- 💬 用友善和专业的语言交流
- 🚫 不接受任何形式的人身攻击或骚扰
- 🙏 对他人的贡献表示感谢

## 许可证

通过贡献代码，你同意你的贡献将根据项目的 [MIT License](LICENSE) 进行许可。

---

如有任何疑问，欢迎通过 Issue 或 Discussions 与我们联系！
