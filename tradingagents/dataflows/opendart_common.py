"""Shared OpenDART (금융감독원 전자공시) helpers: API key, corp-code mapping, calls.

OpenDART is the FSS's official government open API for Korean corporate
disclosures and audited financial statements — the authoritative source that
fills yfinance's thin KR fundamentals. Free key (``DART_API_KEY``) registered
at opendart.fss.or.kr. Uses the existing ``requests`` dependency (no new SDK).

The 6-digit KRX stock code (from a .KS/.KQ ticker) must be mapped to DART's
8-digit ``corp_code`` via the one-shot ``corpCode.xml`` download, which is
cached on disk (it lists every filer, ~3.5 MB) and parsed once per process.
"""

from __future__ import annotations

import io
import logging
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .config import get_config
from .kr_utils import to_krx_code
from .rate_limit import safe_get
from .symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)

_BASE = "https://opendart.fss.or.kr/api"
_CORP_CODE_URL = f"{_BASE}/corpCode.xml"

# Parsed {stock_code(6-digit) -> corp_code(8-digit)} cached for the process.
_CORP_MAP: dict[str, str] | None = None


class OpenDartNotConfiguredError(ValueError):
    """Raised when DART_API_KEY is absent. Subclasses ValueError so
    route_to_vendor's generic except catches it and skips to the next vendor."""


def get_api_key() -> str:
    key = os.environ.get("DART_API_KEY")
    if not key:
        raise OpenDartNotConfiguredError(
            "DART_API_KEY is not set. Register a free key at opendart.fss.or.kr "
            "and add DART_API_KEY=... to your .env."
        )
    return key


def _corp_code_zip_path() -> Path:
    cache_dir = get_config().get("data_cache_dir") or os.path.join(
        os.path.expanduser("~"), ".tradingagents", "cache"
    )
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / "opendart_corpcode.zip"


def _load_corp_map() -> dict[str, str]:
    """Download (once, cached) and parse the corpCode.xml mapping."""
    global _CORP_MAP
    if _CORP_MAP is not None:
        return _CORP_MAP

    zip_path = _corp_code_zip_path()
    if not zip_path.exists():
        resp = safe_get(_CORP_CODE_URL, params={"crtfc_key": get_api_key()}, timeout=30.0)
        zip_path.write_bytes(resp.content)

    mapping: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as z:
        xml_bytes = z.read(z.namelist()[0])
    # XXE / billion-laughs hardening without a new dependency: a legitimate
    # OpenDART corpCode payload carries no DTD, so reject any DOCTYPE/ENTITY
    # declaration before handing the bytes to the stdlib parser (which would
    # otherwise be willing to expand internal entities).
    head = xml_bytes[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        raise ValueError("unexpected DTD/entity declaration in OpenDART corpCode XML")
    for el in ET.fromstring(xml_bytes).iter("list"):
        stock_code = (el.findtext("stock_code") or "").strip()
        corp_code = (el.findtext("corp_code") or "").strip()
        if stock_code and corp_code:
            mapping[stock_code] = corp_code

    _CORP_MAP = mapping
    return mapping


def corp_code_for(ticker: str) -> str:
    """Map a Korean ticker (005930.KS / 247540.KQ / 005930) to a DART corp_code."""
    code = to_krx_code(ticker)  # raises ValueError for non-KR -> dispatcher skips
    corp = _load_corp_map().get(code)
    if not corp:
        raise NoMarketDataError(ticker, code, "no OpenDART corp_code for this KRX code")
    return corp


def dart_get(path: str, **params) -> dict:
    """GET an OpenDART JSON endpoint, injecting the key and checking status.

    OpenDART returns ``status='000'`` on success; ``'013'`` means no data for
    the query, which we surface as NoMarketDataError so the dispatcher can fall
    back to another vendor.
    """
    params["crtfc_key"] = get_api_key()
    resp = safe_get(f"{_BASE}/{path}", params=params, timeout=20.0)
    data = resp.json()
    status = data.get("status")
    if status == "013":  # 조회된 데이터가 없습니다
        raise NoMarketDataError(params.get("corp_code", "?"), None, f"OpenDART: {data.get('message')}")
    if status != "000":
        raise RuntimeError(f"OpenDART error {status}: {data.get('message')}")
    return data
