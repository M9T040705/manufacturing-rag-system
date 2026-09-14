"""
语义分块器
按标题层级 + 固定长度 + overlap 的混合策略分块
针对制造业专业术语优化
"""
import re
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from config.settings import settings


# 制造业专业术语词典（示例，实际可扩展到 2000+ 条）
MANUFACTURING_TERMS = {
    "调质处理", "淬火", "回火", "退火", "正火", "渗碳", "氮化",
    "金相组织", "晶粒大小", "硬度", "抗拉强度", "屈服强度", "伸长率",
    "冲击韧性", "疲劳强度", "蠕变", "应力集中", "残余应力",
    "车削", "铣削", "钻削", "磨削", "镗削", "刨削", "拉削",
    "数控加工", "CNC", "加工中心", "电火花", "线切割", "激光切割",
    "焊接", "电弧焊", "气体保护焊", "氩弧焊", "点焊", "钎焊",
    "铸造", "锻造", "冲压", "注塑", "挤出", "吹塑",
    "公差", "配合", "表面粗糙度", "形位公差", "基准",
    "设备维护", "预防性维护", "预测性维护", "故障诊断", "点检",
    "PLC", "SCADA", "DCS", "传感器", "变频器", "伺服电机",
    "ISO9001", "IATF16949", "六西格玛", "5S", "TPM", "精益生产",
}


class SemanticChunker:
    """语义感知的文档分块器"""

    # 中文标题正则（支持一、二、1. 1.1 等格式）
    HEADING_PATTERNS = [
        r"^[一二三四五六七八九十]+[、.．]\s*.+",       # 一、xxx
        r"^\d+[、.．]\s*.+",                                # 1. xxx
        r"^\d+\.\d+[、.．]?\s*.+",                         # 1.1 xxx
        r"^\d+\.\d+\.\d+[、.．]?\s*.+",                   # 1.1.1 xxx
        r"^第[一二三四五六七八九十]+[章节条款]\s*.+",     # 第一章 xxx
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.document_chunk_size
        self.chunk_overlap = chunk_overlap or settings.document_chunk_overlap
        self._heading_regex = re.compile("|".join(self.HEADING_PATTERNS), re.MULTILINE)
        logger.info(f"分块器初始化: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行分块

        策略：
        1. 先按标题层级切分
        2. 对超过 chunk_size 的段落再按固定长度切分
        3. 保留 overlap 上下文
        4. 专业术语不被切断
        """
        all_chunks = []

        for doc in documents:
            chunks = self._split_single(doc)
            all_chunks.extend(chunks)

        logger.info(f"分块完成: {len(documents)} 个文档 → {len(all_chunks)} 个块")
        return all_chunks

    def _split_single(self, document: Document) -> List[Document]:
        """分块单个文档"""
        text = document.page_content
        metadata = dict(document.metadata)

        # 步骤1：按标题切分
        sections = self._split_by_headings(text)

        # 步骤2：对每个 section 按长度切分
        chunks = []
        for heading, section_text in sections:
            section_chunks = self._split_by_length(section_text, heading)
            for chunk_text in section_chunks:
                chunk_meta = dict(metadata)
                if heading:
                    chunk_meta["heading"] = heading
                chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))

        return chunks

    def _split_by_headings(self, text: str) -> List[tuple]:
        """按标题层级切分文本，返回 [(heading, content), ...]"""
        lines = text.split("\n")
        sections = []
        current_heading = ""
        current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append(line)
                continue

            if self._is_heading(stripped):
                # 保存上一个 section
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines).strip()))
                current_heading = stripped
                current_lines = []
            else:
                current_lines.append(line)

        # 最后一个 section
        if current_lines:
            sections.append((current_heading, "\n".join(current_lines).strip()))

        # 如果没有识别到标题，整体作为一个 section
        if not sections:
            sections.append(("", text.strip()))

        return sections

    def _split_by_length(self, text: str, heading: str = "") -> List[str]:
        """
        按固定长度切分，保留 overlap
        优化：尽量在句子边界切分，避免切断专业术语
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # 尝试在句子边界结束（句号、问号、感叹号、换行）
            if end < text_len:
                sentence_end = self._find_sentence_boundary(text, start, end)
                if sentence_end > start + self.chunk_size * 0.5:
                    end = sentence_end

            # 避免切断专业术语
            end = self._avoid_term_split(text, end)

            chunk = text[start:end].strip()
            if chunk:
                # 如果有标题，把标题加到 chunk 开头增强上下文
                if heading and not chunk.startswith(heading):
                    chunk = f"{heading}\n{chunk}"
                chunks.append(chunk)

            # 移动 start，保留 overlap
            start = end - self.chunk_overlap
            if start >= text_len:
                break

        return chunks

    def _is_heading(self, line: str) -> bool:
        """判断一行是否为标题"""
        # 长度限制：标题通常不超过 50 字
        if len(line) > 50:
            return False
        return bool(self._heading_regex.match(line))

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """在 [start, end] 范围内找最近的句子结束位置"""
        sentence_enders = "。！？!?；;\n"
        # 从 end 往前找
        for i in range(end, start, -1):
            if text[i - 1] in sentence_enders:
                return i
        return end

    def _avoid_term_split(self, text: str, pos: int) -> int:
        """避免在专业术语中间切分"""
        # 检查 pos 前后是否有专业术语被切断
        window = text[max(0, pos - 10):pos + 10]

        for term in MANUFACTURING_TERMS:
            term_pos = window.find(term)
            if term_pos != -1:
                term_start = max(0, pos - 10) + term_pos
                term_end = term_start + len(term)
                # 如果切点在术语中间，移到术语结束位置
                if term_start < pos < term_end:
                    return term_end

        return pos
