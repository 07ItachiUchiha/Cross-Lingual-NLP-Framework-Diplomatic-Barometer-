"""External signals fetcher (news + event context).

This module is OPTIONAL and does not change the pre-RAG diplomatic-document analysis unless explicitly invoked.

Providers supported:
- NewsData.io (requires API key)
- NewsAPI.org (requires API key)
- WorldNewsAPI.com (requires API key)
- GDELT DOC 2.0 (no API key)

Design goals:
- Never hardcode secrets; keys are read from environment variables.
- Emit reproducible artifacts to data/raw (CSV + JSON run report).
- Keep a conservative, normalized schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


@dataclass
class ExternalSignalsConfig:
    query: str
    language: str = "en"
    max_per_provider: int = 100
    timeframe_hours: int = 168  # 7 days for providers that support a rolling window
    # Optional geopolitics-focused filters
    keywords: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    countries: Optional[List[str]] = None  # ISO-ish codes like in, us, jp (provider support varies)


class ExternalSignalsFetcher:
    def __init__(self, output_dir: Optional[str] = None, timeout_seconds: int = 25):
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else project_root / "data" / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timeout_seconds = int(timeout_seconds)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            }
        )

        # Keys are optional; providers without keys are skipped.
        self.newsdata_key = os.getenv("NEWSDATA_API_KEY", "").strip()
        self.newsapi_key = os.getenv("NEWSAPI_API_KEY", "").strip()
        self.worldnews_key = os.getenv("WORLDNEWS_API_KEY", "").strip()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_str(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _split_csv(value: Optional[str]) -> List[str]:
        if not value:
            return []
        out: List[str] = []
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
        return out

    @staticmethod
    def _country_name_hint(code: str) -> str:
        """Minimal mapping used only to enrich queries for providers without country filters."""
        c = str(code or "").strip().lower()
        return {
            "in": "India",
            "us": "United States",
            "jp": "Japan",
            "cn": "China",
            "ru": "Russia",
            "ua": "Ukraine",
            "gb": "United Kingdom",
            "eu": "European Union",
        }.get(c, c)

    def build_query_plain(self, cfg: ExternalSignalsConfig) -> str:
        """Build a simple query string (space-separated tokens).

        Use this for providers that may not support boolean syntax.
        """

        base = self._safe_str(cfg.query)
        keywords = [k.strip() for k in (cfg.keywords or []) if str(k).strip()]
        countries = [c.strip() for c in (cfg.countries or []) if str(c).strip()]

        parts = [base] if base else []
        if keywords:
            parts.append(" ".join(keywords))
        if countries:
            parts.append(" ".join(self._country_name_hint(c) for c in countries))
        return " ".join(p for p in parts if p).strip()

    def build_query_boolean(self, cfg: ExternalSignalsConfig) -> str:
        """Build a boolean-friendly query string using OR-grouping.

        Use this for providers that support boolean operators (NewsAPI, GDELT).
        """

        base = self._safe_str(cfg.query)
        keywords = [k.strip() for k in (cfg.keywords or []) if str(k).strip()]
        countries = [self._country_name_hint(c) for c in (cfg.countries or []) if str(c).strip()]

        groups: List[str] = []
        if base:
            groups.append(f"({base})")
        if keywords:
            groups.append("(" + " OR ".join(f'"{k}"' if " " in k else k for k in keywords) + ")")
        if countries:
            groups.append("(" + " OR ".join(countries) + ")")

        if not groups:
            return ""
        if len(groups) == 1:
            return groups[0]
        return " AND ".join(groups)

    @staticmethod
    def _coerce_dt(value: Any) -> Optional[str]:
        if not value:
            return None
        try:
            dt = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(dt):
                return None
            return dt.isoformat()
        except Exception:
            return None

    def _get_json(self, url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Tuple[int, Any]:
        try:
            r = self.session.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
            status = int(getattr(r, "status_code", 0) or 0)
            try:
                return status, r.json()
            except Exception:
                return status, {"_raw_text": getattr(r, "text", "")}
        except Exception as exc:
            # Provider/network failure should not abort the full run.
            return 0, {"_error": str(exc), "_url": url}

    def fetch_newsdata(self, cfg: ExternalSignalsConfig) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if not self.newsdata_key:
            return [], "missing_api_key"

        # NewsData.io latest endpoint
        # Example: https://newsdata.io/api/1/latest?apikey=...&q=...&language=en&timeframe=48
        url = "https://newsdata.io/api/1/latest"
        params: Dict[str, Any] = {
            "apikey": self.newsdata_key,
            "q": self.build_query_plain(cfg),
            "language": cfg.language,
            "timeframe": int(cfg.timeframe_hours),
        }

        if cfg.categories:
            params["category"] = ",".join([c for c in cfg.categories if str(c).strip()])
        if cfg.countries:
            params["country"] = ",".join([c for c in cfg.countries if str(c).strip()])
        status, payload = self._get_json(url, params=params)
        if status == 422:
            # Some plans/inputs reject combined filters; retry once with relaxed filters.
            relaxed = dict(params)
            relaxed.pop("category", None)
            relaxed.pop("country", None)
            # keep timeframe only if within common documented bounds
            tf = int(relaxed.get("timeframe", 0) or 0)
            if tf and tf > 48:
                relaxed["timeframe"] = 48
            status, payload = self._get_json(url, params=relaxed)
            if status == 422:
                # Still rejected; drop timeframe too.
                relaxed2 = dict(relaxed)
                relaxed2.pop("timeframe", None)
                status, payload = self._get_json(url, params=relaxed2)
        if isinstance(payload, dict) and payload.get("_error"):
            return [], str(payload.get("_error"))
        if status >= 400:
            return [], f"http_{status}"

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            return [], "unexpected_payload"

        items: List[Dict[str, Any]] = []
        for row in results[: cfg.max_per_provider]:
            items.append(
                {
                    "provider": "newsdata",
                    "published_at": self._coerce_dt(row.get("pubDate")),
                    "title": self._safe_str(row.get("title")),
                    "description": self._safe_str(row.get("description")),
                    "url": self._safe_str(row.get("link")),
                    "source_name": self._safe_str(row.get("source_id") or row.get("source")),
                    "language": self._safe_str(row.get("language") or cfg.language),
                    "country": self._safe_str(row.get("country")),
                    "raw": row,
                }
            )
        return items, None

    def fetch_newsapi(self, cfg: ExternalSignalsConfig) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if not self.newsapi_key:
            return [], "missing_api_key"

        # NewsAPI: always use /v2/everything for analysis/discovery.
        # Country/category filters are not supported here; they are incorporated as query hints.
        url = "https://newsapi.org/v2/everything"
        params: Dict[str, Any] = {
            "apiKey": self.newsapi_key,
            "q": self.build_query_boolean(cfg),
            "language": cfg.language,
            "pageSize": min(int(cfg.max_per_provider), 100),
            "sortBy": "publishedAt",
        }
        status, payload = self._get_json(url, params=params)
        if isinstance(payload, dict) and payload.get("_error"):
            return [], str(payload.get("_error"))
        if status >= 400:
            return [], f"http_{status}"

        articles = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(articles, list):
            return [], "unexpected_payload"

        items: List[Dict[str, Any]] = []
        for row in articles[: cfg.max_per_provider]:
            src = row.get("source") if isinstance(row.get("source"), dict) else {}
            items.append(
                {
                    "provider": "newsapi",
                    "published_at": self._coerce_dt(row.get("publishedAt")),
                    "title": self._safe_str(row.get("title")),
                    "description": self._safe_str(row.get("description")),
                    "url": self._safe_str(row.get("url")),
                    "source_name": self._safe_str(src.get("name")),
                    "language": cfg.language,
                    "country": "",
                    "raw": row,
                }
            )
        return items, None

    def fetch_worldnews(self, cfg: ExternalSignalsConfig) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if not self.worldnews_key:
            return [], "missing_api_key"

        # WorldNewsAPI Search News
        # Example: https://api.worldnewsapi.com/search-news?api-key=...&text=tesla
        url = "https://api.worldnewsapi.com/search-news"
        params: Dict[str, Any] = {
            "api-key": self.worldnews_key,
            "text": self.build_query_plain(cfg),
            "language": cfg.language,
            "number": min(int(cfg.max_per_provider), 100),
            "sort": "publish-time",
            "sort-direction": "desc",
        }
        status, payload = self._get_json(url, params=params)
        if isinstance(payload, dict) and payload.get("_error"):
            return [], str(payload.get("_error"))
        if status >= 400:
            return [], f"http_{status}"

        news = payload.get("news") if isinstance(payload, dict) else None
        if not isinstance(news, list):
            return [], "unexpected_payload"

        items: List[Dict[str, Any]] = []
        for row in news[: cfg.max_per_provider]:
            items.append(
                {
                    "provider": "worldnews",
                    "published_at": self._coerce_dt(row.get("publish_date") or row.get("publishDate") or row.get("publish_time")),
                    "title": self._safe_str(row.get("title")),
                    "description": self._safe_str(row.get("text"))[:5000],
                    "url": self._safe_str(row.get("url")),
                    "source_name": self._safe_str(row.get("source_country")) or self._safe_str(row.get("source")),
                    "language": self._safe_str(row.get("language") or cfg.language),
                    "country": self._safe_str(row.get("source_country")),
                    "raw": row,
                }
            )
        return items, None

    def fetch_gdelt(self, cfg: ExternalSignalsConfig, timespan: str = "1week") -> Tuple[List[Dict[str, Any]], Optional[str]]:
        # GDELT DOC 2.0 API (no key)
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params: Dict[str, Any] = {
            "query": self.build_query_boolean(cfg),
            "mode": "artlist",
            "format": "json",
            "maxrecords": min(int(cfg.max_per_provider), 250),
            "timespan": timespan,
            "sort": "datedesc",
        }
        status, payload = self._get_json(url, params=params)
        if isinstance(payload, dict) and payload.get("_error"):
            return [], str(payload.get("_error"))
        if status >= 400:
            return [], f"http_{status}"

        articles = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(articles, list):
            return [], "unexpected_payload"

        items: List[Dict[str, Any]] = []
        for row in articles[: cfg.max_per_provider]:
            items.append(
                {
                    "provider": "gdelt",
                    "published_at": self._coerce_dt(row.get("seendate") or row.get("datetime") or row.get("date")),
                    "title": self._safe_str(row.get("title")),
                    "description": self._safe_str(row.get("sourceCountry")) + " " + self._safe_str(row.get("sourceCollection")),
                    "url": self._safe_str(row.get("url")),
                    "source_name": self._safe_str(row.get("sourceCommonName") or row.get("domain")),
                    "language": self._safe_str(row.get("language")),
                    "country": self._safe_str(row.get("sourceCountry")),
                    "raw": row,
                }
            )
        return items, None

    def fetch_all(self, cfg: ExternalSignalsConfig) -> Dict[str, Any]:
        run_started = self._utc_now_iso()

        items: List[Dict[str, Any]] = []
        provider_counts: Dict[str, int] = {}
        provider_status: Dict[str, Dict[str, Any]] = {}

        def _call(provider: str, fn) -> None:
            try:
                provider_items, err = fn(cfg)
            except Exception as exc:  # pragma: no cover
                provider_items, err = [], str(exc)

            provider_counts[provider] = int(len(provider_items))
            provider_status[provider] = {
                "returned": int(len(provider_items)),
                "error": err,
            }

            if provider_items:
                items.extend(provider_items)

        _call("newsdata", self.fetch_newsdata)
        _call("newsapi", self.fetch_newsapi)
        _call("worldnews", self.fetch_worldnews)
        _call("gdelt", self.fetch_gdelt)

        df = pd.DataFrame(items)
        if not df.empty:
            # Stable column order
            cols = [
                "provider",
                "published_at",
                "title",
                "description",
                "url",
                "source_name",
                "language",
                "country",
                "raw",
            ]
            for col in cols:
                if col not in df.columns:
                    df[col] = ""
            df = df[cols]

        report = {
            "run_started_utc": run_started,
            "run_finished_utc": self._utc_now_iso(),
            "query": cfg.query,
            "built_query": {
                "plain": self.build_query_plain(cfg),
                "boolean": self.build_query_boolean(cfg),
            },
            "language": cfg.language,
            "timeframe_hours": int(cfg.timeframe_hours),
            "max_per_provider": int(cfg.max_per_provider),
            "provider_counts": provider_counts,
            "provider_status": provider_status,
            "total_items": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
            "outputs": {},
        }

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"external_signals_{stamp}.csv"
        json_path = self.output_dir / f"external_signals_report_{stamp}.json"

        if not df.empty:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            report["outputs"]["signals_csv"] = str(csv_path)

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        report["outputs"]["report_json"] = str(json_path)

        return {"df": df, "report": report}
