"""
系统评测脚本
用法：python scripts/run_evaluation.py --eval-set data/eval_set.json --output results/eval_report.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.evaluation import RAGEvaluator


async def chat_fn(question: str):
    """调用 API 进行问答（示例实现）"""
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/api/chat",
            json={"question": question, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        sources = [s["source"] for s in data.get("sources", [])]
        return data["answer"], sources


def main():
    parser = argparse.ArgumentParser(description="RAG 系统评测脚本")
    parser.add_argument("--eval-set", type=str, required=True, help="评测集 JSON 文件路径")
    parser.add_argument("--output", type=str, default="results/eval_report.json", help="评测报告输出路径")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API 地址")
    args = parser.parse_args()

    logger.info(f"加载评测集: {args.eval_set}")

    # 初始化评测器
    evaluator = RAGEvaluator(chat_fn=chat_fn)

    # 加载评测集
    eval_items = evaluator.load_eval_set(args.eval_set)

    # 执行评测
    report = asyncio.run(evaluator.evaluate(eval_items, output_path=args.output))

    # 打印报告
    print("\n" + "=" * 60)
    print("评测报告")
    print("=" * 60)
    print(f"总测试数:     {report['total']}")
    print(f"通过数:       {report['passed']}")
    print(f"准确率:       {report['accuracy']:.2%}")
    print(f"检索命中率:   {report['retrieval_hit_rate']:.2%}")
    print(f"平均相关性:   {report['avg_relevance']:.2%}")
    print(f"平均延迟:     {report['avg_latency_ms']} ms")
    print(f"P95 延迟:     {report['p95_latency_ms']} ms")
    print("=" * 60)

    if report.get("failed_cases"):
        print("\n失败案例 (前10条):")
        for i, case in enumerate(report["failed_cases"], 1):
            print(f"  {i}. [{case['reason']}] {case['question'][:50]}...")

    logger.info(f"评测报告已保存: {args.output}")


if __name__ == "__main__":
    main()
