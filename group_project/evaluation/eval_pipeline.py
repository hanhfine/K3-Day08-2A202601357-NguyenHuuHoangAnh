"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

TOP_K = 5

# 2 configs so sánh A/B: Config A dùng full pipeline Task 9 (hybrid + RRF + rerank),
# Config B chỉ dùng Semantic Search đơn thuần (bỏ BM25/RRF/rerank) để đo lợi ích thật
# sự của phần hybrid mang lại.
CONFIG_DESCRIPTIONS = {
    "hybrid_rerank": "Hybrid Search (Semantic + BM25 qua RRF) + Rerank + PageIndex fallback (Task 9 đầy đủ)",
    "dense_only": "Dense-only: chỉ Semantic Search (Cosine Similarity), không BM25/RRF/rerank",
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _retrieve_hybrid(query: str, top_k: int) -> list[dict]:
    from src.task9_retrieval_pipeline import retrieve
    return retrieve(query, top_k=top_k, use_reranking=True)


def _retrieve_dense_only(query: str, top_k: int) -> list[dict]:
    from src.task5_semantic_search import semantic_search
    return semantic_search(query, top_k=top_k)


RETRIEVAL_FNS = {
    "hybrid_rerank": _retrieve_hybrid,
    "dense_only": _retrieve_dense_only,
}


def _generate_for_config(query: str, top_k: int, retrieval_fn) -> dict:
    """
    Sinh câu trả lời cho 1 config cụ thể, tái dùng reorder/format/LLM client
    từ Task 10 nhưng đổi bước retrieval để so sánh A/B công bằng.
    """
    from src.task10_generation import (
        reorder_for_llm, format_context, _get_llm_client,
        SYSTEM_PROMPT, TEMPERATURE, TOP_P,
    )

    chunks = retrieval_fn(query, top_k)
    if not chunks:
        return {"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có", "sources": []}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    client, model = _get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return {"answer": response.choices[0].message.content, "sources": chunks}


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(eval_data: dict) -> dict:
    """
    Evaluate 1 bộ generations (question/answer/contexts/ground_truth) bằng RAGAS.

    Khác với chữ ký gợi ý ban đầu (rag_pipeline, golden_dataset): hàm này nhận thẳng
    eval_data đã sinh sẵn, để compare_configs() có thể tái sử dụng cho nhiều config
    retrieval khác nhau (hybrid vs dense-only) mà không cần 1 "rag_pipeline" cố định.

    pip install ragas
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    df = result.to_pandas()
    return {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "per_question": df,
    }


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="UniversityServices_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict], top_k: int = TOP_K) -> dict:
    """
    So sánh A/B giữa 2 configs:
    - Config A "hybrid_rerank": Semantic + BM25 qua RRF + Rerank (Task 9 đầy đủ)
    - Config B "dense_only": chỉ Semantic Search, không BM25/RRF/rerank

    Trả về dict: config_name -> kết quả evaluate_with_ragas() (bao gồm cả điểm
    trung bình lẫn per_question DataFrame để export_results() dùng phân tích
    worst performers).
    """
    results = {}
    for config_name, retrieval_fn in RETRIEVAL_FNS.items():
        print(f"\n=== Running config: {config_name} ===")
        eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        for item in golden_dataset:
            print(f"  - {item['question'][:60]}...")
            gen = _generate_for_config(item["question"], top_k, retrieval_fn)
            contexts = [c["content"] for c in gen["sources"]]
            eval_data["question"].append(item["question"])
            eval_data["answer"].append(gen["answer"])
            eval_data["contexts"].append(contexts if contexts else ["(no context retrieved)"])
            eval_data["ground_truth"].append(item["expected_answer"])

        results[config_name] = evaluate_with_ragas(eval_data)
    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(comparison: dict) -> None:
    """Export A/B comparison results (từ compare_configs) ra results.md"""
    config_names = list(comparison.keys())
    a, b = config_names[0], config_names[1]
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    lines = ["# RAG Evaluation Results\n"]
    lines.append("## Framework sử dụng\n")
    lines.append("RAGAS (`pip install ragas`) — LLM judge: OpenAI `gpt-4o-mini` (qua `OPENAI_API_KEY`).\n")
    lines.append("---\n")
    lines.append("## Overall Scores\n")
    lines.append(f"| Metric | Config A ({a}) | Config B ({b}) | Δ |")
    lines.append("|--------|---------------------------|----------------------|---|")

    scores_a, scores_b = [], []
    for m in metrics:
        va, vb = comparison[a][m], comparison[b][m]
        scores_a.append(va)
        scores_b.append(vb)
        lines.append(f"| {metric_labels[m]} | {va:.3f} | {vb:.3f} | {va - vb:+.3f} |")
    mean_a, mean_b = sum(scores_a) / len(scores_a), sum(scores_b) / len(scores_b)
    lines.append(f"| **Average** | **{mean_a:.3f}** | **{mean_b:.3f}** | **{mean_a - mean_b:+.3f}** |")
    lines.append("\n---\n")

    lines.append("## A/B Comparison Analysis\n")
    lines.append(f"**Config A ({a}):**")
    lines.append(f"> {CONFIG_DESCRIPTIONS[a]}\n")
    lines.append(f"**Config B ({b}):**")
    lines.append(f"> {CONFIG_DESCRIPTIONS[b]}\n")
    winner = a if mean_a >= mean_b else b
    reason = (
        "Kết hợp BM25 + RRF + rerank giúp bù các trường hợp Semantic Search bỏ sót từ khoá/số hiệu chính xác "
        "(vd tên chính sách, số tiền, mã điều khoản)."
        if winner == a
        else "Dense-only đơn giản hơn nhưng vẫn đủ tốt cho corpus nhỏ này; phần hybrid thêm vào chưa mang lại "
        "lợi ích tương xứng với độ phức tạp."
    )
    lines.append("**Kết luận:**")
    lines.append(
        f"> Config **{winner}** đạt điểm trung bình cao hơn ({max(mean_a, mean_b):.3f} so với "
        f"{min(mean_a, mean_b):.3f}). {reason}"
    )
    lines.append("\n---\n")

    df_a = comparison[a]["per_question"]
    df_a = df_a.assign(avg_score=df_a[metrics].mean(axis=1)).sort_values("avg_score").head(3)
    lines.append("## Worst Performers (Bottom 3, theo Config A)\n")
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |")
    lines.append("|---|----------|-------------|-----------|--------|---------------|------------|")
    for i, (_, row) in enumerate(df_a.iterrows(), 1):
        stage = "Retrieval" if row["context_recall"] < 0.5 else "Generation"
        cause = (
            "Corpus thiếu evidence liên quan hoặc threshold fallback chưa kích hoạt đúng lúc"
            if stage == "Retrieval"
            else "LLM diễn giải context chưa sát hoặc trích dẫn nguồn chưa đầy đủ"
        )
        q = str(row["question"])[:50]
        lines.append(
            f"| {i} | {q}... | {row['faithfulness']:.2f} | {row['answer_relevancy']:.2f} | "
            f"{row['context_recall']:.2f} | {stage} | {cause} |"
        )
    lines.append("\n---\n")

    lines.append("## Recommendations\n")
    lines.append("### Cải tiến 1")
    lines.append("**Action:** Mở rộng corpus (thêm tài liệu legal/news) cho các chủ đề có Context Recall thấp.")
    lines.append("**Expected impact:** Tăng Context Recall & Faithfulness vì retriever có nhiều evidence hơn để chọn.\n")
    lines.append("### Cải tiến 2")
    lines.append("**Action:** Calibrate lại `SCORE_THRESHOLD` ở Task 9 dựa trên phân phối điểm cosine thực tế khi corpus mở rộng.")
    lines.append("**Expected impact:** Kích hoạt PageIndex fallback đúng lúc hơn, giảm câu trả lời rác khi hybrid search yếu.\n")
    lines.append("### Cải tiến 3")
    lines.append("**Action:** Thử embedding model mạnh hơn (vd `BAAI/bge-m3`) nếu hạ tầng cho phép tải model lớn hơn.")
    lines.append("**Expected impact:** Tăng độ chính xác Semantic Search, đặc biệt với câu hỏi diễn đạt khác từ ngữ trong tài liệu gốc.\n")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Exported results to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    comparison = compare_configs(golden_dataset, top_k=TOP_K)
    export_results(comparison)
