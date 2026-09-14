"""系统集成模块：OA对接、数据同步"""
from .oa_client import OAClient
from .data_sync import DataSyncManager

__all__ = ["OAClient", "DataSyncManager"]
