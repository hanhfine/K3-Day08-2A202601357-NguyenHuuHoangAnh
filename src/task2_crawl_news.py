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
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://www.rmit.edu.vn/students/my-studies/fees-and-payments",
    "https://www.rmit.edu.vn/students/my-studies/fees-and-payments/tuition-fees-faq-and-support",
    "https://www.rmit.edu.vn/students/my-studies/international-students/accommodation-for-international-students",
    "https://www.rmit.edu.vn/study-at-rmit/scholarships/future-undergraduate-student-scholarships",
    "https://www.rmit.edu.vn/students/support/library-services",
]


class HTMLTextExtractor(HTMLParser):
    """Chuyển HTML công khai thành text Markdown đơn giản."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.title_parts = []
        self.skip_depth = 0
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
        if tag == "title":
            self.in_title = True
        if not self.skip_depth:
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self.parts.append("\n" + "#" * int(tag[1]) + " ")
            elif tag == "li":
                self.parts.append("\n- ")
            elif tag in {"p", "div", "section", "article", "main", "br"}:
                self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and (
            tag in {"p", "div", "section", "article", "main", "li"}
            or tag in {"h1", "h2", "h3", "h4", "h5", "h6"}
        ):
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
        self.parts.append(data)

    @property
    def title(self):
        return " ".join("".join(self.title_parts).split())

    @property
    def markdown(self):
        lines = []
        for line in "".join(self.parts).splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line and (not lines or line != lines[-1]):
                lines.append(line)
        return "\n\n".join(lines)


def crawl_article_http(url: str) -> dict:
    """Fallback HTTP crawler không cần package ngoài."""
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "vi,en;q=0.9",
        },
    )
    with urlopen(request, timeout=45) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(charset, errors="replace")

    parser = HTMLTextExtractor()
    parser.feed(html)
    content = parser.markdown
    if len(content) < 500:
        raise ValueError(f"Nội dung quá ngắn ({len(content)} ký tự): {url}")

    return {
        "url": url,
        "title": parser.title or url,
        "date_crawled": datetime.now().astimezone().isoformat(),
        "content_markdown": content,
    }


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
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return await asyncio.to_thread(crawl_article_http, url)

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        if not result.success:
            raise RuntimeError(result.error_message)

        markdown = result.markdown
        if not isinstance(markdown, str):
            markdown = (
                getattr(markdown, "fit_markdown", "")
                or getattr(markdown, "raw_markdown", "")
            )
        if len(markdown.strip()) < 500:
            raise ValueError(f"Nội dung quá ngắn ({len(markdown)} ký tự): {url}")

        return {
            "url": url,
            "title": (result.metadata or {}).get("title") or url,
            "date_crawled": datetime.now().astimezone().isoformat(),
            "content_markdown": markdown.strip(),
        }
    except Exception as error:
        print(f"  [WARN] Crawl4AI chưa sẵn sàng ({error}); dùng HTTP crawler")
        return await asyncio.to_thread(crawl_article_http, url)


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [OK] Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
