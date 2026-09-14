"""
RAG 系统评测器
评估指标：检索准确率、答案相关性、响应延迟、引用覆盖率
支持批量评测集，输出详细报告
"""
import json
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel


class EvalItem(BaseModel):
    """评测项"""
    question: str
    expected_answer: str
    expected_sources: List[str] = []
    category: str = "general"


class EvalResult(BaseModel):
    """单条评测结果"""
    question: str
    answer: str
    expected_answer: str
    sources: List[str]
    retrieval_hit: bool
    answer_relevance: float
    latency_ms: int
    passed: bool


class RAGEvaluator:
    """RAG 系统评测器"""

    def __init__(self, chat_fn, retrieval_fn=None):
        """
        Args:
            chat_fn: 问答函数，输入 question 返回 (answer, sources, latency)
            retrieval_fn: 检索函数，输入 question 返回 sources
        """
        self.chat_fn = chat_fn
        self.retrieval_fn = retrieval_fn

    def load_eval_set(self, file_path: str) -> List[EvalItem]:
        """加载评测集（JSON 格式）"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"评测集不存在: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = []
        for item in data:
            items.append(EvalItem(**item))

        logger.info(f"加载评测集: {len(items)} 条")
        return items

    async def evaluate(
        self,
        eval_items: List[EvalItem],
        output_path: Optional[str] = None,
    ) -> dict:
        """
        执行批量评测

        Args:
            eval_items: 评测项列表
            output_path: 结果输出路径

        Returns:
            评测报告 dict
        """
        results: List[EvalResult] = []
        total = len(eval_items)

        logger.info(f"开始评测，共 {total} 条")

        for i, item in enumerate(eval_items, 1):
            try:
                start = time.time()
                answer, sources = await self.chat_fn(item.question)
                latency = int((time.time() - start) * 1000)

                # 检索命中率
                retrieval_hit = self._check_retrieval_hit(sources, item.expected_sources)

                # 答案相关性（简单的关键词重叠度）
                relevance = self._calc_relevance(answer, item.expected_answer)

                # 是否通过（相关性 > 0.6 且检索命中）
                passed = relevance >= 0.6 and (not item.expected_sources or retrieval_hit)

                result = EvalResult(
                    question=item.question,
                    answer=answer,
                    expected_answer=item.expected_answer,
                    sources=sources,
                    retrieval_hit=retrieval_hit,
                    answer_relevance=relevance,
                    latency_ms=latency,
                    passed=passed,
                )
                results.append(result)

                if i % 10 == 0:
                    logger.info(f"评测进度: {i}/{total}")

            except Exception as e:
                logger.error(f"评测失败 [{i}] {item.question}: {e}")
                results.append(EvalResult(
                    question=item.question,
                    answer=f"ERROR: {e}",
                    expected_answer=item.expected_answer,
                    sources=[],
                    retrieval_hit=False,
                    answer_relevance=0.0,
                    latency_ms=0,
                    passed=False,
                ))

        # 汇总指标
        report = self._generate_report(results)

        # 保存结果
        if output_path:
            self._save_results(results, report, output_path)

        logger.info(f"评测完成: 准确率={report['accuracy']:.2%}, 平均延迟={report['avg_latency_ms']}ms")
        return report

    def _check_retrieval_hit(self, sources: List[str], expected: List[str]) -> bool:
        """检查检索结果是否包含期望来源"""
        if not expected:
            return True  # 没有期望来源时默认命中
        for exp in expected:
            for src in sources:
                if exp in src or src in exp:
                    return True
        return False

    def _calc_relevance(self, answer: str, expected: str) -> float:
        """
        计算答案相关性（基于关键词重叠）
        实际生产中可替换为 LLM-as-Judge
        """
        if not answer or not expected:
            return 0.0

        # 简单分词（字符级）
        answer_chars = set(answer)
        expected_chars = set(expected)

        if not expected_chars:
            return 0.0

        overlap = answer_chars & expected_chars
        return len(overlap) / len(expected_chars)

    def _generate_report(self, results: List[EvalResult]) -> dict:
        """生成汇总评测报告"""
        total = len(results)
        if total == 0:
            return {}

        passed = sum(1 for r in results if r.passed)
        retrieval_hits = sum(1 for r in results if r.retrieval_hit)
        avg_relevance = sum(r.answer_relevance for r in results) / total
        avg_latency = sum(r.latency_ms for r in results) / total
        p95_latency = sorted(r.latency_ms for r in results)[int(total * 0.95) - 1] if total >= 20 else max(r.latency_ms for r in results)

        # 按类别统计
        category_stats = {}
        for r in results:
            cat = r.question[:10]  # 简化分类
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0}
            category_stats[cat]["total"] += 1
            if r.passed:
                category_stats[cat]["passed"] += 1

        return {
            "total": total,
            "passed": passed,
            "accuracy": passed / total,
            "retrieval_hit_rate": retrieval_hits / total,
            "avg_relevance": avg_relevance,
            "avg_latency_ms": int(avg_latency),
            "p95_latency_ms": int(p95_latency),
            "failed_cases": [
                {"question": r.question, "reason": "相关性低" if r.answer_relevance < 0.6 else "检索未命中"}
                for r in results if not r.passed
            ][:10],
        }

    def _save_results(self, results: List[EvalResult], report: dict, output_path: str):
        """保存评测结果到文件"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "report": report,
            "details": [r.model_dump() for r in results],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"评测结果已保存: {output_path}")
