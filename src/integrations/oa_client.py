"""
OA 系统客户端
对接企业 OA 系统，实现文档拉取、SSO 认证、操作日志审计
"""
from typing import List, Optional

import httpx
from loguru import logger

from config.settings import settings


class OAClient:
    """OA 系统 API 客户端"""

    def __init__(self):
        self.base_url = settings.oa_api_base.rstrip("/")
        self.token = settings.oa_api_token
        self.client = httpx.Client(timeout=30.0)
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_documents(self, last_sync_time: Optional[str] = None, limit: int = 100) -> List[dict]:
        """
        获取 OA 系统中的文档列表

        Args:
            last_sync_time: 上次同步时间，只获取之后更新的文档
            limit: 每页数量

        Returns:
            文档列表，每个包含 id, title, url, update_time, department 等
        """
        params = {"limit": limit}
        if last_sync_time:
            params["update_time__gte"] = last_sync_time

        try:
            response = self.client.get(
                f"{self.base_url}/documents",
                params=params,
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
            docs = data.get("results", data.get("data", []))
            logger.info(f"从 OA 获取 {len(docs)} 个文档")
            return docs
        except Exception as e:
            logger.error(f"OA 文档获取失败: {e}")
            raise

    def download_document(self, doc_id: str, save_path: str) -> bool:
        """
        下载 OA 文档到本地

        Args:
            doc_id: 文档 ID
            save_path: 本地保存路径

        Returns:
            是否下载成功
        """
        try:
            response = self.client.get(
                f"{self.base_url}/documents/{doc_id}/download",
                headers=self.headers,
            )
            response.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(response.content)

            logger.info(f"文档下载成功: {doc_id} -> {save_path}")
            return True
        except Exception as e:
            logger.error(f"文档下载失败 {doc_id}: {e}")
            return False

    def verify_token(self, token: str) -> Optional[dict]:
        """
        验证 SSO Token，获取用户信息

        Args:
            token: 用户登录 Token

        Returns:
            用户信息 dict，验证失败返回 None
        """
        try:
            response = self.client.get(
                f"{self.base_url}/auth/verify",
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.warning(f"SSO Token 验证失败: {e}")
            return None

    def write_audit_log(self, user_id: str, action: str, detail: str) -> bool:
        """
        写入操作审计日志

        Args:
            user_id: 用户 ID
            action: 操作类型
            detail: 操作详情

        Returns:
            是否成功
        """
        try:
            payload = {
                "user_id": user_id,
                "action": action,
                "detail": detail,
                "module": "rag_system",
            }
            response = self.client.post(
                f"{self.base_url}/audit/logs",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"审计日志写入失败: {e}")
            return False

    def close(self):
        """关闭客户端"""
        self.client.close()
