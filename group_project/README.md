# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**

- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**

```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework                                           | Cài đặt               | Đặc điểm                                      |
| --------------------------------------------------- | ------------------------ | ------------------------------------------------- |
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas`    | Chuẩn industry cho RAG eval, 3 trục chính      |
| [TruLens](https://github.com/truera/trulens)         | `pip install trulens`  | Dashboard UI, feedback functions mạnh            |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
data/landing/{legal,news}/  (PDF, DOCX, JSON)
        │  Task 1-2: thu thập
        ▼
data/standardized/*.md      (Markdown chuẩn hoá)
        │  Task 3: convert
        ▼
┌───────────────────────────────────────────────────────────┐
│ Task 4: Chunking (800/100) + Embedding                     │
│ (paraphrase-multilingual-MiniLM-L12-v2, 384-dim)            │
│              → ChromaDB (chroma_db/)                       │
└───────────────────────────────────────────────────────────┘
        │
        ▼
   User query (app.py / eval_pipeline.py)
        │
        ├──► Task 5: Semantic Search (Cosine, dense) ─┐
        │                                              ├─► Task 7: RRF Merge
        └──► Task 6: Lexical Search (BM25, sparse) ────┘         (k=60)
                                                              │
                                                              ▼
                                          Task 9: retrieve()
                                          - Rerank (RRF) → top_k
                                          - Nếu best cosine < 0.35
                                            → Task 8: PageIndex Fallback
                                              (vectorless, structural)
                                                              │
                                                              ▼
                              Task 10: generate_with_citation()
                              - reorder_for_llm (front + back[::-1])
                              - format_context (kèm source)
                              - LLM (OpenRouter/OpenAI) → answer + citation
                                                              │
                                                              ▼
                        app.py (Streamlit Chatbot) — hiển thị answer + sources
                        group_project/evaluation/eval_pipeline.py — RAGAS A/B eval
```

---

## Phân Công Công Việc

| Thành viên            | MSSV        | Nhiệm vụ (Role, Phương Án A) | Trạng thái |
| ----------------------- | ----------- | ---------- | ------------ |
| Nguyễn Cao Nam         | 2A202601377 | Role 3 — Frontend & Chatbot Dev: Task 8 (PageIndex), `app.py` Streamlit, Task 10 (Generation) | ✅ Hoàn thành |
| Nguyễn Phương Nam    | 2A202601952 | Role 2 — Data & Retrieval Specialist: Task 1-3 (thu thập & convert dữ liệu), Task 4 (ChromaDB), Task 5 (Semantic Search), Task 7 (RRF Rerank), Task 9 (Retrieval Pipeline) | ✅ Hoàn thành |
| Lương Trung Chiến    | 2A202601391 | Role 4 — Evaluation & QA: Task 6 (BM25), `golden_dataset.json`, `eval_pipeline.py` (RAGAS), `results.md` | ✅ Hoàn thành |
| Nguyễn Hữu Hoàng Anh | 2A202601357 | Role 1 — Team Leader & RAG Architect: điều phối tiến độ, kiểm tra tham số chunking/RRF, tổng hợp `supervisor.py` & pipeline | ✅ Hoàn thành |

**Ghi chú tiến độ:** `pytest tests/test_individual.py` đạt **35/35 passed** (CP4). CP5: `app.py` chạy được (đã test `streamlit run`, kết nối `generate_with_citation()`), `eval_pipeline.py` đã chạy RAGAS A/B thật (17 câu hỏi, 2 configs `hybrid_rerank` vs `dense_only`) và xuất `results.md`.

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
