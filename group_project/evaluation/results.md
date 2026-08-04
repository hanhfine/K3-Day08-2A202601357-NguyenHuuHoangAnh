# RAG Evaluation Results

## Framework sử dụng

RAGAS (`pip install ragas`) — LLM judge: OpenAI `gpt-4o-mini` (qua `OPENAI_API_KEY`).

---

## Overall Scores

| Metric | Config A (hybrid_rerank) | Config B (dense_only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.365 | 0.403 | -0.038 |
| Answer Relevance | 0.349 | 0.252 | +0.097 |
| Context Recall | 0.765 | 0.618 | +0.147 |
| Context Precision | 0.785 | 0.782 | +0.004 |
| **Average** | **0.566** | **0.514** | **+0.053** |

---

## A/B Comparison Analysis

**Config A (hybrid_rerank):**
> Hybrid Search (Semantic + BM25 qua RRF) + Rerank + PageIndex fallback (Task 9 đầy đủ)

**Config B (dense_only):**
> Dense-only: chỉ Semantic Search (Cosine Similarity), không BM25/RRF/rerank

**Kết luận:**
> Config **hybrid_rerank** đạt điểm trung bình cao hơn (0.566 so với 0.514). Kết hợp BM25 + RRF + rerank giúp bù các trường hợp Semantic Search bỏ sót từ khoá/số hiệu chính xác (vd tên chính sách, số tiền, mã điều khoản).

---

## Worst Performers (Bottom 3, theo Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Một sinh viên có thể nhận tối đa bao nhiêu học bổn... | 0.00 | 0.00 | 0.00 | Retrieval | Corpus thiếu evidence liên quan hoặc threshold fallback chưa kích hoạt đúng lúc |
| 2 | Sinh viên nhận học bổng cần duy trì tối thiểu bao ... | 0.00 | 0.00 | 0.00 | Retrieval | Corpus thiếu evidence liên quan hoặc threshold fallback chưa kích hoạt đúng lúc |
| 3 | Học phí hàng năm của chương trình Business tại RMI... | 0.00 | 0.00 | 0.00 | Retrieval | Corpus thiếu evidence liên quan hoặc threshold fallback chưa kích hoạt đúng lúc |

---

## Recommendations

### Cải tiến 1
**Action:** Mở rộng corpus (thêm tài liệu legal/news) cho các chủ đề có Context Recall thấp.
**Expected impact:** Tăng Context Recall & Faithfulness vì retriever có nhiều evidence hơn để chọn.

### Cải tiến 2
**Action:** Calibrate lại `SCORE_THRESHOLD` ở Task 9 dựa trên phân phối điểm cosine thực tế khi corpus mở rộng.
**Expected impact:** Kích hoạt PageIndex fallback đúng lúc hơn, giảm câu trả lời rác khi hybrid search yếu.

### Cải tiến 3
**Action:** Thử embedding model mạnh hơn (vd `BAAI/bge-m3`) nếu hạ tầng cho phép tải model lớn hơn.
**Expected impact:** Tăng độ chính xác Semantic Search, đặc biệt với câu hỏi diễn đạt khác từ ngữ trong tài liệu gốc.
