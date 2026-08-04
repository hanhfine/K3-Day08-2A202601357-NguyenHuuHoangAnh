"""Task 1 - Thu thap van ban chinh sach cua RMIT Vietnam.

Script tai ba tai lieu PDF cong khai ve hoc phi, hoc bong va cho o vao
``data/landing/legal``. Chay tu thu muc goc cua du an bang lenh:

    python src/task1_collect_legal_docs.py
"""

from pathlib import Path
from typing import Final

import requests


DATA_DIR: Final = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"
REQUEST_TIMEOUT: Final = 60
MIN_FILE_SIZE: Final = 1024

# Cac URL deu tro den tai lieu PDF cong khai tren website chinh thuc cua
# RMIT Vietnam. Ten file khong dau va mo ta ro noi dung tai lieu.
DOCUMENTS: Final = (
    {
        "topic": "Hoc phi va cac khoan phi phu thu",
        "url": (
            "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/"
            "documents/pdfs/study-at-rmit/tuition-fees/"
            "student-fees-and-charges-guide-06-2026.pdf"
        ),
        "filename": "student-fees-and-charges-guide-rmit-2026.pdf",
    },
    {
        "topic": "Dieu khoan va dieu kien hoc bong",
        "url": (
            "https://www.rmit.edu.vn/content/dam/rmit/vn/en/"
            "assets-for-production/documents/pdfs/study-at-rmit/"
            "scholarships/english-pdf/"
            "rmit-university-vietnam-scholarship-terms-and-conditions.pdf"
        ),
        "filename": "scholarship-terms-and-conditions-rmit.pdf",
    },
    {
        "topic": "Huong dan cho o cho sinh vien quoc te",
        "url": (
            "https://www.rmit.edu.vn/content/dam/rmit/vn/en/"
            "assets-for-production/documents/pdfs/students/accommodation/"
            "accommodation-advice-for-international-students-in-vietnam.pdf"
        ),
        "filename": "accommodation-advice-rmit-vietnam.pdf",
    },
)


def setup_directory() -> None:
    """Tao thu muc dich neu thu muc chua ton tai."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc da san sang: {DATA_DIR}")


def is_valid_pdf(filepath: Path) -> bool:
    """Kiem tra kich thuoc toi thieu va chu ky dau file PDF."""
    if not filepath.is_file() or filepath.stat().st_size <= MIN_FILE_SIZE:
        return False

    with filepath.open("rb") as file:
        return file.read(5) == b"%PDF-"


def download_file(url: str, filename: str) -> Path:
    """Tai mot PDF, kiem tra noi dung va luu an toan vao DATA_DIR.

    File duoc tai vao duoi ``.part`` truoc. Chi sau khi qua kiem tra PDF,
    file tam moi duoc doi ten thanh file dich de tranh giu lai file loi.
    """
    destination = DATA_DIR / filename
    temporary = destination.with_suffix(destination.suffix + ".part")

    if is_valid_pdf(destination):
        print(f"[SKIP] File da ton tai va hop le: {destination.name}")
        return destination

    headers = {"User-Agent": "Mozilla/5.0 (RAG-Lab-Document-Collector/1.0)"}

    try:
        with requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            stream=True,
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for block in response.iter_content(chunk_size=64 * 1024):
                    if block:
                        file.write(block)

        if not is_valid_pdf(temporary):
            raise ValueError(f"Noi dung tai ve khong phai PDF hop le: {url}")

        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"[OK] Da tai {destination.name} ({destination.stat().st_size:,} bytes)")
    return destination


def collect_legal_documents() -> list[Path]:
    """Tai toan bo tai lieu cau hinh trong DOCUMENTS."""
    setup_directory()
    downloaded_files = []

    for document in DOCUMENTS:
        print(f"\nDang xu ly: {document['topic']}")
        downloaded_files.append(
            download_file(document["url"], document["filename"])
        )

    print(f"\nHoan thanh Task 1: {len(downloaded_files)} tai lieu hop le.")
    return downloaded_files


if __name__ == "__main__":
    collect_legal_documents()
