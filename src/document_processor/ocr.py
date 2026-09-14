"""
OCR 文字识别处理器
基于 PaddleOCR，支持中文识别、GPU 加速
"""
from typing import Optional

from loguru import logger

from config.settings import settings


class OCRProcessor:
    """PaddleOCR 封装"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._ocr = None
        self._initialized = True

    def _get_ocr(self):
        """懒加载 OCR 模型"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=settings.ocr_lang,
                    use_gpu=settings.ocr_use_gpu,
                    show_log=False,
                )
                logger.info("PaddleOCR 模型加载成功")
            except Exception as e:
                logger.error(f"PaddleOCR 加载失败: {e}")
                raise
        return self._ocr

    def recognize_file(self, file_path: str) -> str:
        """识别图片文件中的文字"""
        ocr = self._get_ocr()
        result = ocr.ocr(file_path, cls=True)
        return self._parse_result(result)

    def recognize_bytes(self, image_bytes: bytes) -> str:
        """识别图片字节流中的文字"""
        import numpy as np
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)

        ocr = self._get_ocr()
        result = ocr.ocr(image_array, cls=True)
        return self._parse_result(result)

    def _parse_result(self, result) -> str:
        """解析 OCR 结果为纯文本"""
        if not result or not result[0]:
            return ""

        texts = []
        for line in result[0]:
            if line and len(line) >= 2:
                text_info = line[1]
                if text_info and len(text_info) >= 1:
                    texts.append(text_info[0])

        return "\n".join(texts)
