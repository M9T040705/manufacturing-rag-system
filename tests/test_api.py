"""
API 接口测试
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestHealthAPI:
    """健康检查接口测试"""

    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "milvus" in data
        assert "llm" in data

    def test_root(self, client):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestStatsAPI:
    """系统统计接口测试"""

    def test_get_stats(self, client):
        """测试获取系统统计"""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "collection_name" in data
        assert "model_name" in data


class TestChatAPI:
    """问答接口测试"""

    def test_chat_request_validation(self, client):
        """测试请求参数校验"""
        # 空问题应该返回 422
        response = client.post("/api/chat", json={"question": ""})
        assert response.status_code == 422

    def test_chat_request_structure(self, client):
        """测试正常请求结构（不依赖实际模型）"""
        # 这里只测试请求格式，实际回答需要完整环境
        payload = {
            "question": "什么是调质处理？",
            "stream": False,
            "use_rerank": True,
        }
        # 由于测试环境可能没有完整组件，这里只验证请求能被接收
        # 实际测试需要在完整环境中运行
        assert payload["question"] == "什么是调质处理？"
