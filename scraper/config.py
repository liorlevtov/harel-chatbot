"""Scraper configuration settings."""

BASE_URL = "https://www.harel-group.co.il"

# Insurance domains to scrape
INSURANCE_DOMAINS = [
    "car",
    "life",
    "travel",
    "health",
    "dental",
    "mortgage",
    "business",
    "apartment",
]

# Starting URLs (constant instead of function)
START_URLS = [f"{BASE_URL}/insurance/{domain}" for domain in INSURANCE_DOMAINS]

# Sitemap URL
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"

# Concurrency settings
MAX_CONCURRENT_REQUESTS = 20
CONNECTION_TIMEOUT = 30  # seconds
READ_TIMEOUT = 60  # seconds

# Streaming chunk size for large file downloads
CHUNK_SIZE = 8192  # 8KB

# Output directory
OUTPUT_DIR = "data"

# URL patterns to include (must contain these)
INCLUDE_PATTERNS = [
    "/insurance/",
    "/Insurance/",
    "/Policies/",
    "/policies/",
]

# URL patterns to exclude
EXCLUDE_PATTERNS = [
    "/login",
    "/Login",
    "/personal-zone",
    "/agent",
    "/Agent",
    "javascript:",
    "mailto:",
    "tel:",
    "#",
]

# Headers for requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Domain keywords for URL categorization
DOMAIN_KEYWORDS = {
    "car": ["car", "vehicle", "auto", "רכב", "zmh", "צמה"],
    "life": ["life", "חיים"],
    "travel": ["travel", "נסיעות", "darkon", "passport", "first-class", "firstclass"],
    "health": ["health", "בריאות", "medical", "רפואה"],
    "dental": ["dental", "שיניים"],
    "mortgage": ["mortgage", "משכנתא"],
    "business": ["business", "עסק", "עסקי", "moshav", "מושב"],
    "apartment": ["apartment", "דירה", "home", "house", "adira", "אדירה"],
}


def is_pdf_url(url: str, content_type: str = "") -> bool:
    """Check if URL or content type indicates a PDF file."""
    return "pdf" in content_type.lower() or url.lower().endswith(".pdf")


def categorize_url(url: str) -> str:
    """Determine which insurance domain a URL belongs to."""
    url_lower = url.lower()

    # Direct domain match (highest priority)
    for domain in INSURANCE_DOMAINS:
        if f"/insurance/{domain}" in url_lower:
            return domain

    # Secondary path match
    for domain in INSURANCE_DOMAINS:
        if f"/{domain}/" in url_lower:
            return domain

    # Keyword matching (lowest priority)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in url_lower:
                return domain

    return "general"


def clean_url(url: str) -> str:
    """Clean URL by removing fragments and normalizing."""
    from urllib.parse import urlparse

    parsed = urlparse(url)

    # Ensure path is not empty
    path = parsed.path or "/"

    # Reconstruct URL without fragment
    clean = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        clean += f"?{parsed.query}"

    return clean
