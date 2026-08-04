"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PROJECT_ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
LANDING_DIR = PROJECT_ROOT / "data" / "landing"
DOC_IDS_PATH = PROJECT_ROOT / "pageindex_doc_ids.json"
PDF_CACHE_DIR = PROJECT_ROOT / "pageindex_pdfs"

POLL_INTERVAL_SECONDS = 2.0
POLL_TIMEOUT_SECONDS = 120.0


def _get_client():
    """Tạo PageIndex client và không bao giờ log API key."""
    api_key = os.getenv("PAGEINDEX_API_KEY") or PAGEINDEX_API_KEY
    if not api_key:
        raise RuntimeError(
            "Chưa có PAGEINDEX_API_KEY. Hãy thêm key vào file .env."
        )

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=api_key)


def _load_document_registry() -> dict:
    """Đọc mapping source -> metadata/doc_id đã upload."""
    if not DOC_IDS_PATH.exists():
        return {}

    try:
        data = json.loads(DOC_IDS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Không đọc được {DOC_IDS_PATH.name}: {error}") from error

    # Hỗ trợ cả schema mới và mapping source -> doc_id đơn giản.
    documents = data.get("documents", data) if isinstance(data, dict) else {}
    registry = {}
    for source, entry in documents.items():
        if isinstance(entry, str):
            registry[source] = {"doc_id": entry, "source": Path(source).name}
        elif isinstance(entry, dict) and entry.get("doc_id"):
            registry[source] = entry
    return registry


def _save_document_registry(registry: dict) -> None:
    """Ghi registry atomically để tránh file JSON dở dang."""
    payload = {"version": 1, "documents": registry}
    temp_path = DOC_IDS_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(DOC_IDS_PATH)


def _find_unicode_font() -> Path:
    """Tìm font Unicode phổ biến trên Windows, Linux hoặc macOS."""
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return font_path
    raise RuntimeError(
        "Không tìm thấy font Unicode để chuyển Markdown sang PDF."
    )


def _markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> Path:
    """Chuyển Markdown thành PDF Unicode đơn giản cho PageIndex Cloud."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    text = markdown_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Markdown rỗng: {markdown_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("PageIndexUnicode", fname=str(_find_unicode_font()))
    pdf.set_font("PageIndexUnicode", size=10)

    for raw_line in text.splitlines():
        # Giữ nội dung heading nhưng bỏ ký hiệu Markdown ở đầu.
        line = raw_line.lstrip("#").strip() if raw_line.startswith("#") else raw_line
        pdf.multi_cell(
            w=0,
            h=5,
            text=line or " ",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            wrapmode="CHAR",
        )

    pdf.output(str(pdf_path))
    return pdf_path


def _prepare_pdf(markdown_path: Path) -> Path:
    """Dùng PDF gốc nếu có; nếu không thì tạo PDF từ Markdown."""
    relative = markdown_path.relative_to(STANDARDIZED_DIR)
    original_pdf = LANDING_DIR / relative.with_suffix(".pdf")
    if original_pdf.exists():
        return original_pdf

    cached_pdf = PDF_CACHE_DIR / relative.with_suffix(".pdf")
    if (
        not cached_pdf.exists()
        or cached_pdf.stat().st_mtime < markdown_path.stat().st_mtime
    ):
        _markdown_to_pdf(markdown_path, cached_pdf)
    return cached_pdf


def upload_documents(force: bool = False) -> dict:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    markdown_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not markdown_files:
        raise RuntimeError(
            "Không có Markdown trong data/standardized. Hãy chạy Task 3 trước."
        )

    client = _get_client()
    registry = _load_document_registry()

    for markdown_path in markdown_files:
        relative_source = markdown_path.relative_to(STANDARDIZED_DIR).as_posix()
        cached = registry.get(relative_source, {})
        if cached.get("doc_id") and not force:
            print(f"  [SKIP] {relative_source} -> {cached['doc_id']}")
            continue

        pdf_path = _prepare_pdf(markdown_path)
        response = client.submit_document(str(pdf_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(
                f"PageIndex không trả doc_id cho {relative_source}: {response}"
            )

        registry[relative_source] = {
            "doc_id": doc_id,
            "source": markdown_path.name,
            "type": markdown_path.parent.name,
            "pdf_path": str(pdf_path.relative_to(PROJECT_ROOT)),
        }
        # Lưu sau từng upload để không mất các doc_id đã thành công.
        _save_document_registry(registry)
        print(f"  [OK] Uploaded: {relative_source} -> {doc_id}")

    return registry


def _poll_retrieval(
    client,
    retrieval_id: str,
    timeout_seconds: float = POLL_TIMEOUT_SECONDS,
    interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> dict:
    """Poll retrieval cho đến khi có kết quả, thất bại hoặc timeout."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = client.get_retrieval(retrieval_id)
        payload = response.get("result", response)
        if not isinstance(payload, dict):
            payload = {}
        status = str(
            response.get("status") or payload.get("status") or ""
        ).lower()
        retrieved_nodes = (
            response.get("retrieved_nodes")
            or payload.get("retrieved_nodes")
        )
        if retrieved_nodes or status in {
            "completed",
            "complete",
            "succeeded",
            "success",
            "ready",
        }:
            return response
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(
                f"PageIndex retrieval {retrieval_id} thất bại: {response}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"PageIndex retrieval {retrieval_id} quá {timeout_seconds:.0f}s"
            )
        time.sleep(interval_seconds)


def _iter_relevant_items(value):
    """Duyệt schema relevant_contents lồng nhiều cấp của API retrieval."""
    if isinstance(value, list):
        for item in value:
            yield from _iter_relevant_items(item)
        return

    if not isinstance(value, dict):
        return

    content = (
        value.get("relevant_content")
        or value.get("content")
        or value.get("text")
    )
    if isinstance(content, str) and content.strip():
        yield value

    for key in ("relevant_contents", "contents", "nodes", "children"):
        if key in value:
            yield from _iter_relevant_items(value[key])


def _parse_retrieval(
    response: dict,
    registry_entry: dict,
    top_k: int,
) -> list[dict]:
    """Chuẩn hóa PageIndex response về schema chung của retrieval pipeline."""
    payload = response.get("result", response)
    nodes = payload.get("retrieved_nodes", []) if isinstance(payload, dict) else []
    results = []
    seen = set()

    for rank, item in enumerate(_iter_relevant_items(nodes), start=1):
        content = (
            item.get("relevant_content")
            or item.get("content")
            or item.get("text")
            or ""
        ).strip()
        section = (
            item.get("section_title")
            or item.get("title")
            or item.get("section")
            or "Unknown section"
        )
        dedup_key = (content, section)
        if not content or dedup_key in seen:
            continue
        seen.add(dedup_key)

        results.append(
            {
                "content": content,
                # Retrieval API không trả similarity; score được suy ra theo rank.
                "score": round(1.0 / rank, 6),
                "metadata": {
                    "source": registry_entry.get("source", "Unknown"),
                    "type": registry_entry.get("type", "unknown"),
                    "section": section,
                    "doc_id": registry_entry.get("doc_id", ""),
                    "node_id": item.get("node_id", ""),
                },
                "source": "pageindex",
            }
        )
        if len(results) >= top_k:
            break
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not isinstance(query, str):
        raise TypeError("query phải là chuỗi")
    query = query.strip()
    if not query or top_k <= 0:
        return []

    # PageIndex là fallback tùy chọn: thiếu key hoặc chưa upload thì trả []
    # để Task 9 có thể tiếp tục dùng kết quả hybrid thay vì crash.
    if not (os.getenv("PAGEINDEX_API_KEY") or PAGEINDEX_API_KEY):
        return []

    registry = _load_document_registry()
    if not registry:
        print("  [WARN] Chưa có pageindex_doc_ids.json; hãy chạy upload_documents()")
        return []

    client = _get_client()
    combined = []
    for relative_source, entry in registry.items():
        doc_id = entry.get("doc_id")
        if not doc_id:
            continue

        registry_entry = {
            **entry,
            "source": entry.get("source") or Path(relative_source).name,
            "doc_id": doc_id,
        }
        try:
            if hasattr(client, "is_retrieval_ready") and not client.is_retrieval_ready(doc_id):
                print(f"  [SKIP] PageIndex document chưa sẵn sàng: {relative_source}")
                continue

            submitted = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
            if not retrieval_id:
                raise RuntimeError(
                    f"PageIndex không trả retrieval_id: {submitted}"
                )

            response = _poll_retrieval(client, retrieval_id)
            combined.extend(
                _parse_retrieval(response, registry_entry, top_k=top_k)
            )
        except Exception as error:
            # Một document lỗi không được làm hỏng toàn bộ fallback.
            print(f"  [WARN] PageIndex query lỗi với {relative_source}: {error}")

    # Gộp nhiều document, giữ điểm cao nhất cho nội dung trùng nhau.
    deduplicated = {}
    for item in combined:
        key = item["content"]
        previous = deduplicated.get(key)
        if previous is None or item["score"] > previous["score"]:
            deduplicated[key] = item

    results = sorted(
        deduplicated.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
