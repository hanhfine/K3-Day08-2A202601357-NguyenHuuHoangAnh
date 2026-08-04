"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách 5 URL từ RMIT Vietnam cần crawl
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/students/support/it-support",
    "https://www.rmit.edu.vn/students/support/student-connect",
    "https://www.rmit.edu.vn/students/my-studies/new-students/enrol-as-a-new-student",
    "https://www.rmit.edu.vn/student-life/life-and-work-opportunities/career-ready-hub",
    "https://www.rmit.edu.vn/study-at-rmit/international-students/international-student-support-services",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(
        word_count_threshold=10,       # bỏ qua block quá ngắn (nav, footer)
        excluded_tags=["nav", "footer", "header", "script", "style"],
        exclude_external_links=True,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

        title = (
            result.metadata.get("title", "")
            if result.metadata
            else ""
        ) or url.rstrip("/").split("/")[-1]

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)

            # Dùng slug từ URL làm tên file cho dễ nhận biết
            slug = url.rstrip("/").split("/")[-1][:40]
            filename = f"article_{i:02d}_{slug}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            chars = len(article.get("content_markdown", ""))
            print(f"  ✓ Saved: {filepath}  ({chars} chars)")
        except Exception as e:
            print(f"  ✗ Lỗi khi crawl {url}: {e}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
