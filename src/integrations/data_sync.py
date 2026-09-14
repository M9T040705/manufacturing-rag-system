"""
数据同步管理器
定时从 OA 系统同步增量文档，自动解析入库
支持断点续传、失败重试、同步状态记录
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config.settings import settings
from .oa_client import OAClient


class DataSyncManager:
    """OA 文档增量同步管理器"""

    def __init__(
        self,
        oa_client: OAClient,
        ingest_callback,  # 文档入库回调函数
        sync_dir: str = "data/raw/sync",
        state_file: str = "data/sync_state.json",
    ):
        self.oa_client = oa_client
        self.ingest_callback = ingest_callback
        self.sync_dir = Path(sync_dir)
        self.state_file = Path(state_file)
        self.scheduler: Optional[BackgroundScheduler] = None

        self.sync_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict:
        """加载同步状态"""
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"last_sync_time": None, "last_sync_count": 0, "total_synced": 0}

    def _save_state(self, state: dict):
        """保存同步状态"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def sync_once(self) -> dict:
        """
        执行一次增量同步

        Returns:
            同步结果统计
        """
        logger.info("开始执行数据同步...")
        state = self._load_state()
        last_sync_time = state.get("last_sync_time")

        try:
            # 1. 从 OA 获取增量文档
            docs = self.oa_client.get_documents(
                last_sync_time=last_sync_time,
                limit=settings.oa_sync_batch_size,
            )

            if not docs:
                logger.info("没有需要同步的新文档")
                state["last_sync_time"] = datetime.now().isoformat()
                self._save_state(state)
                return {"synced": 0, "skipped": 0, "failed": 0}

            synced = 0
            failed = 0

            # 2. 逐个下载并入库
            for doc in docs:
                try:
                    doc_id = doc.get("id")
                    filename = doc.get("title", f"doc_{doc_id}")
                    file_ext = os.path.splitext(filename)[1] or ".pdf"
                    save_path = self.sync_dir / f"{doc_id}{file_ext}"

                    # 下载
                    success = self.oa_client.download_document(doc_id, str(save_path))
                    if not success:
                        failed += 1
                        continue

                    # 入库（调用回调）
                    self.ingest_callback(str(save_path), doc)
                    synced += 1

                    # 清理已入库的文件
                    # os.unlink(save_path)

                except Exception as e:
                    logger.error(f"文档同步失败 {doc.get('id')}: {e}")
                    failed += 1

            # 3. 更新状态
            state["last_sync_time"] = datetime.now().isoformat()
            state["last_sync_count"] = synced
            state["total_synced"] = state.get("total_synced", 0) + synced
            self._save_state(state)

            result = {"synced": synced, "skipped": 0, "failed": failed}
            logger.info(f"数据同步完成: {result}")
            return result

        except Exception as e:
            logger.error(f"数据同步失败: {e}")
            raise

    def start_scheduler(self):
        """启动定时同步调度器"""
        if not settings.oa_sync_enabled:
            logger.info("OA 同步未启用")
            return

        self.scheduler = BackgroundScheduler()

        # 解析 cron 表达式
        cron_parts = settings.oa_sync_cron.split()
        if len(cron_parts) == 5:
            trigger = CronTrigger(
                minute=cron_parts[0],
                hour=cron_parts[1],
                day=cron_parts[2],
                month=cron_parts[3],
                day_of_week=cron_parts[4],
            )
        else:
            trigger = CronTrigger(hour=2, minute=0)  # 默认每天凌晨2点

        self.scheduler.add_job(
            self.sync_once,
            trigger=trigger,
            id="oa_data_sync",
            name="OA文档增量同步",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info(f"定时同步已启动: {settings.oa_sync_cron}")

    def stop_scheduler(self):
        """停止调度器"""
        if self.scheduler:
            self.scheduler.shutdown()
            logger.info("定时同步已停止")
