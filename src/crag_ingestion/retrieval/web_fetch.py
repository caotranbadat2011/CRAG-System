from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler, ProxyHandler, Request, build_opener,
)

from ..config import WebSearchConfig


class WebPageError(ValueError):
    """A search result could not be used as safe textual evidence."""


@dataclass(frozen=True, slots=True)
class FetchedPage:
    url: str
    text: str
    content_sha256: str
    fetched_at: str
    truncated: bool


def normalize_public_url(url: str, allowed_domains: tuple[str, ...] = ()) -> str:
    """Reject non-web, credentialed, local, and disallowed URL targets."""
    try:
        parsed = urlsplit(url.strip())
        host = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise WebPageError("Invalid URL") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not host or parsed.username is not None or parsed.password is not None
        or any(ord(char) < 32 for char in url)
    ):
        raise WebPageError("Only public HTTP(S) URLs without credentials are allowed")
    if port not in (None, 80, 443):
        raise WebPageError("Nonstandard web ports are not allowed")
    if "%" in host:
        raise WebPageError("Local or encoded hostnames are not allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or host.casefold() == "localhost":
            raise WebPageError("Local hostnames are not allowed") from None
        try:
            ascii_host = host.encode("idna").decode("ascii").lower().rstrip(".")
        except UnicodeError:
            raise WebPageError("Invalid web hostname") from None
        if ascii_host == "localhost" or ascii_host.endswith(
            (".local", ".localhost", ".internal")
        ):
            raise WebPageError("Local hostnames are not allowed")
    else:
        if not address.is_global:
            raise WebPageError("Nonpublic IP addresses are not allowed")
        ascii_host = f"[{host}]" if address.version == 6 else host
    if allowed_domains:
        plain_host = ascii_host.strip("[]")
        allowed = tuple(domain.casefold().lstrip(".") for domain in allowed_domains)
        if not any(plain_host == domain or plain_host.endswith("." + domain) for domain in allowed):
            raise WebPageError("Source domain is not in the allowlist")
    netloc = ascii_host + (f":{port}" if port else "")
    # HTTP request targets must be ASCII. Keep existing percent escapes so
    # URLs returned by search providers are not encoded twice.
    path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
    return urlunsplit((parsed.scheme.lower(), netloc, path, query, ""))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: object, code: int,
                         msg: str, headers: object, newurl: str) -> None:
        return None


class _PageTextParser(HTMLParser):
    _ignored = {"script", "style", "nav", "footer", "header", "aside", "form", "svg", "noscript"}
    _blocks = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._ignored:
            self.ignored_depth += 1
        elif self.ignored_depth:
            return
        elif tag in self._blocks:
            self.parts.append("\n\n")
        elif tag in {"br", "td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._ignored and self.ignored_depth:
            self.ignored_depth -= 1
        elif self.ignored_depth:
            return
        elif tag in self._blocks:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        return re.sub(r"\n{3,}", "\n\n", value).strip()


class SafePageFetcher:
    """Fetch bounded public HTML/text pages, rejecting redirects to private hosts."""

    def __init__(self, config: WebSearchConfig | None = None) -> None:
        self.config = config or WebSearchConfig()
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def fetch(self, url: str) -> FetchedPage:
        target = normalize_public_url(url, self.config.allowed_domains)
        for _ in range(4):
            self._check_dns(target)
            request = Request(
                target,
                headers={"User-Agent": "CRAG-System/0.1 (+text-only research fetch)",
                         "Accept": "text/html,text/plain"},
            )
            try:
                response = self._opener.open(request, timeout=self.config.fetch_timeout)
            except HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308}:
                    location = exc.headers.get("Location")
                    if not location:
                        raise WebPageError("Redirect has no destination") from None
                    target = normalize_public_url(
                        urljoin(target, location), self.config.allowed_domains
                    )
                    continue
                raise WebPageError(f"Web page returned HTTP {exc.code}") from None
            except (URLError, TimeoutError, OSError):
                raise WebPageError("Web page could not be fetched") from None
            with response:
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "text/plain"}:
                    raise WebPageError("Web page is not HTML or plain text")
                raw = response.read(self.config.max_page_bytes + 1)
                if len(raw) > self.config.max_page_bytes:
                    raise WebPageError("Web page exceeds the byte limit")
                charset = response.headers.get_content_charset() or "utf-8"
                try:
                    decoded = raw.decode(charset, errors="replace")
                except LookupError:
                    decoded = raw.decode("utf-8", errors="replace")
            if content_type == "text/html":
                parser = _PageTextParser()
                parser.feed(decoded)
                decoded = parser.text()
            else:
                decoded = re.sub(r"[ \t]+", " ", decoded).strip()
            truncated = len(decoded) > self.config.max_page_chars
            text = decoded[: self.config.max_page_chars].strip()
            if not text:
                raise WebPageError("Web page has no extractable text")
            return FetchedPage(
                target, text, hashlib.sha256(text.encode("utf-8")).hexdigest(),
                datetime.now(timezone.utc).isoformat(), truncated,
            )
        raise WebPageError("Web page redirected too many times")

    @staticmethod
    def _check_dns(url: str) -> None:
        host = urlsplit(url).hostname
        assert host is not None
        try:
            addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except OSError:
            raise WebPageError("Web host could not be resolved") from None
        if not addresses or any(
            not ipaddress.ip_address(item[4][0]).is_global for item in addresses
        ):
            raise WebPageError("Web host resolves to a nonpublic IP address")
