from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import pandas as pd
import requests

from utils.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _norm_title(value: object) -> str:
    txt = str(value or "").lower().strip()
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"[^a-z0-9\s\-]", "", txt)
    return txt.strip()


def _norm_source(value: object) -> str:
    src = str(value or "").upper().strip()
    alias = {
        "MOFA_ARCHIVE": "MOFA",
        "EMBJPIN": "EMBJPIN",
    }
    return alias.get(src, src)


def _norm_date(value: object) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.date().isoformat()


def _year_from_date_iso(value: str) -> Optional[int]:
    try:
        return int(str(value)[:4])
    except Exception:
        return None


def _is_valid_url(value: object) -> bool:
    u = str(value or "").strip()
    if not u:
        return False
    return u.startswith("http://") or u.startswith("https://")


def _lookup_url_for_row(source: str, title: str) -> str:
    src = _norm_source(source)
    query = str(title or "").strip()
    if not query:
        return ""

    domain_hint = {
        "MEA": "mea.gov.in",
        "MOFA": "mofa.go.jp",
        "MOFA_ARCHIVE": "mofa.go.jp",
        "JETRO": "jetro.go.jp",
        "EMBJPIN": "in.emb-japan.go.jp",
    }.get(src, "")

    if domain_hint:
        return f"https://www.bing.com/search?q={quote_plus(f'site:{domain_hint} {query}') }"
    return f"https://www.bing.com/search?q={quote_plus(query)}"


def _domain_hints_for_source(source: str) -> List[str]:
    src = _norm_source(source)
    hints = {
        "MEA": ["mea.gov.in"],
        "MOFA": ["mofa.go.jp"],
        "MOFA_ARCHIVE": ["mofa.go.jp"],
        "JETRO": ["jetro.go.jp"],
        "EMBJPIN": ["in.emb-japan.go.jp", "emb-japan.go.jp"],
    }
    return hints.get(src, [])


def _token_set(title_norm: str) -> set:
    parts = [p for p in re.split(r"\s+", title_norm) if p and len(p) >= 4]
    return set(parts)


@dataclass(frozen=True)
class BackfillConfig:
    canonical_csv: Path = RAW_DATA_DIR / "india_japan_documents_canonical.csv"
    min_confidence: float = 0.92
    max_live_files: int = 30


@dataclass(frozen=True)
class AssistedBackfillConfig:
    canonical_csv: Path = RAW_DATA_DIR / "india_japan_documents_canonical.csv"
    review_csv: Optional[Path] = None
    min_confidence: float = 0.82
    max_rows: int = 200
    request_timeout: int = 20


@dataclass(frozen=True)
class CuratedArchiveBackfillConfig:
    canonical_csv: Path = RAW_DATA_DIR / "india_japan_documents_canonical.csv"
    review_csv: Optional[Path] = None
    min_confidence: float = 0.86
    max_rows: int = 300
    request_timeout: int = 20


@dataclass(frozen=True)
class SemiAutoApprovalConfig:
    curated_review_csv: Optional[Path] = None
    canonical_csv: Path = RAW_DATA_DIR / "india_japan_documents_canonical.csv"
    medium_min_confidence: float = 0.70
    recommend_approve_confidence: float = 0.82


def _collect_reference_rows(cfg: BackfillConfig) -> pd.DataFrame:
    files: List[Path] = []

    base_primary = RAW_DATA_DIR / "india_japan_documents.csv"
    if base_primary.exists():
        files.append(base_primary)

    if cfg.canonical_csv.exists():
        files.append(cfg.canonical_csv)

    live = sorted(RAW_DATA_DIR.glob("live_scrape_*.csv"), reverse=True)
    files.extend(live[: max(1, int(cfg.max_live_files))])

    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        needed = {"date", "title", "source", "url"}
        if not needed.issubset(set(df.columns)):
            continue

        w = df[["date", "title", "source", "url"]].copy()
        w["_file"] = fp.name
        w = w[w["url"].map(_is_valid_url)].copy()
        if len(w) == 0:
            continue

        frames.append(w)

    if not frames:
        return pd.DataFrame(columns=["date", "title", "source", "url", "_file", "_k_date", "_k_title", "_k_source", "_k_year", "_k_tokens"])

    refs = pd.concat(frames, ignore_index=True)
    refs["_k_date"] = refs["date"].map(_norm_date)
    refs["_k_title"] = refs["title"].map(_norm_title)
    refs["_k_source"] = refs["source"].map(_norm_source)
    refs["_k_year"] = refs["_k_date"].map(_year_from_date_iso)
    refs["_k_tokens"] = refs["_k_title"].map(_token_set)

    refs = refs[(refs["_k_title"].astype(str).str.len() > 0) & (refs["_k_source"].astype(str).str.len() > 0)].copy()
    refs = refs.drop_duplicates(subset=["_k_date", "_k_title", "_k_source", "url"], keep="first")
    return refs


def _choose_candidate(row: pd.Series, refs: pd.DataFrame) -> Tuple[str, float, str, str]:
    k_date = str(row.get("_k_date", ""))
    k_title = str(row.get("_k_title", ""))
    k_source = str(row.get("_k_source", ""))
    k_year = row.get("_k_year", None)
    k_tokens = row.get("_k_tokens", set())

    if not k_title or not k_source:
        return "", 0.0, "none", "Missing normalized title/source"

    exact = refs[
        (refs["_k_date"] == k_date)
        & (refs["_k_title"] == k_title)
        & (refs["_k_source"] == k_source)
    ]
    if len(exact) > 0:
        return str(exact.iloc[0]["url"]), 0.99, "exact_date_title_source", "Exact match on date+title+source"

    ts = refs[(refs["_k_title"] == k_title) & (refs["_k_source"] == k_source)]
    ts_urls = sorted(set(ts["url"].astype(str).tolist()))
    if len(ts_urls) == 1:
        return ts_urls[0], 0.96, "unique_title_source", "Unique URL for title+source"

    same_source = refs[refs["_k_source"] == k_source].copy()
    if len(same_source) == 0:
        return "", 0.0, "none", "No source-aligned references"

    if pd.notna(k_year):
        same_year = same_source[same_source["_k_year"] == k_year].copy()
        if len(same_year) > 0:
            same_source = same_year

    best_ratio = 0.0
    second_ratio = 0.0
    best_url = ""
    best_overlap = 0.0

    for _, cand in same_source.iterrows():
        cand_title = str(cand.get("_k_title", ""))
        cand_url = str(cand.get("url", "")).strip()
        if not cand_title or not cand_url:
            continue

        ratio = SequenceMatcher(None, k_title, cand_title).ratio()
        cand_tokens = cand.get("_k_tokens", set())
        if not isinstance(cand_tokens, set):
            cand_tokens = set()

        overlap = 0.0
        if k_tokens and cand_tokens:
            overlap = len(k_tokens.intersection(cand_tokens)) / max(1, len(k_tokens.union(cand_tokens)))

        score = (0.75 * ratio) + (0.25 * overlap)
        if score > best_ratio:
            second_ratio = best_ratio
            best_ratio = score
            best_url = cand_url
            best_overlap = overlap
        elif score > second_ratio:
            second_ratio = score

    if best_ratio >= 0.93 and (best_ratio - second_ratio) >= 0.03:
        conf = min(0.95, best_ratio)
        return best_url, float(conf), "fuzzy_source_title", f"Fuzzy title match score={best_ratio:.3f}, token_overlap={best_overlap:.3f}"

    return "", 0.0, "none", "No safe candidate above confidence threshold"


def generate_url_backfill_patch(cfg: Optional[BackfillConfig] = None) -> Dict:
    cfg = cfg or BackfillConfig()
    if not cfg.canonical_csv.exists():
        raise FileNotFoundError(f"Canonical corpus not found: {cfg.canonical_csv}")

    canonical_df = pd.read_csv(cfg.canonical_csv)
    if not {"date", "title", "source"}.issubset(set(canonical_df.columns)):
        raise ValueError("Canonical corpus must include date, title, source columns")

    if "url" not in canonical_df.columns:
        canonical_df["url"] = ""

    refs = _collect_reference_rows(cfg)

    working = canonical_df.copy()
    working["_k_date"] = working["date"].map(_norm_date)
    working["_k_title"] = working["title"].map(_norm_title)
    working["_k_source"] = working["source"].map(_norm_source)
    working["_k_year"] = working["_k_date"].map(_year_from_date_iso)
    working["_k_tokens"] = working["_k_title"].map(_token_set)

    working["_url_missing"] = ~working["url"].map(_is_valid_url)
    missing = working[working["_url_missing"]].copy()

    candidate_rows: List[Dict] = []
    patch_rows: List[Dict] = []

    for idx, row in missing.iterrows():
        proposed_url, confidence, method, rationale = _choose_candidate(row, refs)
        entry = {
            "row_index": int(idx),
            "date": row.get("date", ""),
            "source": row.get("source", ""),
            "title": row.get("title", ""),
            "current_url": row.get("url", ""),
            "proposed_url": proposed_url,
            "lookup_url": _lookup_url_for_row(str(row.get("source", "")), str(row.get("title", ""))),
            "confidence": round(float(confidence), 3),
            "method": method,
            "rationale": rationale,
            "review_status": "PENDING",
        }
        candidate_rows.append(entry)

        if proposed_url and confidence >= float(cfg.min_confidence):
            patch_rows.append(entry)

    stamp = _utc_stamp()
    review_csv = PROCESSED_DATA_DIR / f"url_backfill_review_{stamp}.csv"
    patch_csv = PROCESSED_DATA_DIR / f"url_backfill_patch_{stamp}.csv"
    preview_csv = PROCESSED_DATA_DIR / f"url_backfill_preview_canonical_{stamp}.csv"
    report_json = PROCESSED_DATA_DIR / f"url_backfill_report_{stamp}.json"

    review_df = pd.DataFrame(candidate_rows)
    patch_df = pd.DataFrame(patch_rows)
    review_df.to_csv(review_csv, index=False, encoding="utf-8")
    patch_df.to_csv(patch_csv, index=False, encoding="utf-8")

    preview_df = canonical_df.copy()
    if len(patch_df) > 0:
        for _, p in patch_df.iterrows():
            preview_df.at[int(p["row_index"]), "url"] = str(p.get("proposed_url", "")).strip()
    preview_df.to_csv(preview_csv, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_csv": str(cfg.canonical_csv),
        "reference_rows": int(len(refs)),
        "missing_url_rows": int(len(missing)),
        "candidates_generated": int(len(review_df)),
        "proposed_patch_rows": int(len(patch_df)),
        "min_confidence": float(cfg.min_confidence),
        "outputs": {
            "review_csv": str(review_csv),
            "patch_csv": str(patch_csv),
            "preview_canonical_csv": str(preview_csv),
            "report_json": str(report_json),
        },
    }

    with open(report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def _latest_review_csv() -> Optional[Path]:
    files = sorted(PROCESSED_DATA_DIR.glob("url_backfill_review_*.csv"), reverse=True)
    return files[0] if files else None


def _extract_ddg_links(html_text: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_text, flags=re.IGNORECASE | re.DOTALL):
        href = str(m.group(1) or "").strip()
        title_html = str(m.group(2) or "")
        title_text = re.sub(r"<[^>]+>", " ", title_html)
        title_text = html.unescape(re.sub(r"\s+", " ", title_text)).strip()

        if not href:
            continue

        url = href
        if "duckduckgo.com/l/?" in href:
            try:
                q = parse_qs(urlparse(href).query)
                if q.get("uddg"):
                    url = unquote(q["uddg"][0])
            except Exception:
                pass

        if _is_valid_url(url):
            rows.append((url, title_text))

    dedup: List[Tuple[str, str]] = []
    seen = set()
    for u, t in rows:
        if u in seen:
            continue
        seen.add(u)
        dedup.append((u, t))
    return dedup


def _extract_links_from_lookup_page(lookup_url: str, source: str, timeout: int = 20) -> List[Tuple[str, str]]:
    if not _is_valid_url(lookup_url):
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(lookup_url, headers=headers, timeout=int(timeout), allow_redirects=True)
        if int(resp.status_code) != 200:
            return []
        html_text = str(resp.text or "")
    except Exception:
        return []

    raw_urls = re.findall(r"https?://[^\"\'<>\s]+", html_text)
    if not raw_urls:
        return []

    domain_hints = _domain_hints_for_source(source)
    out: List[Tuple[str, str]] = []
    seen = set()
    for u in raw_urls:
        try:
            u = html.unescape(u).strip()
        except Exception:
            continue
        if not _is_valid_url(u):
            continue
        host = urlparse(u).netloc.lower()
        if domain_hints and not any(h in host for h in domain_hints):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append((u, u))

    return out


def _fetch_search_candidates(title: str, source: str, timeout: int = 20, lookup_url: str = "") -> List[Tuple[str, str]]:
    direct = _extract_links_from_lookup_page(lookup_url=lookup_url, source=source, timeout=timeout)
    if direct:
        return direct

    domain_hints = _domain_hints_for_source(source)
    query_text = str(title or "").strip()
    if not query_text:
        return []

    query = query_text
    if domain_hints:
        query = f"site:{domain_hints[0]} {query_text}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get("https://duckduckgo.com/html/", params={"q": query}, headers=headers, timeout=int(timeout))
        if int(resp.status_code) != 200:
            return []
        return _extract_ddg_links(resp.text)
    except Exception:
        return []


def _score_web_candidate(row_title: str, row_date: object, row_source: str, cand_url: str, cand_title: str) -> Tuple[float, str]:
    target = _norm_title(row_title)
    cand_t = _norm_title(cand_title)
    sim = SequenceMatcher(None, target, cand_t).ratio() if target and cand_t else 0.0

    parsed = urlparse(cand_url)
    netloc = parsed.netloc.lower()
    domain_hints = _domain_hints_for_source(row_source)
    domain_ok = any(h in netloc for h in domain_hints) if domain_hints else True
    domain_score = 1.0 if domain_ok else 0.0

    year = _year_from_date_iso(_norm_date(row_date) or "")
    year_hit = 0.0
    if year is not None:
        y = str(year)
        if y in cand_url or y in cand_title:
            year_hit = 1.0

    confidence = (0.55 * sim) + (0.30 * domain_score) + (0.15 * year_hit)
    rationale = f"sim={sim:.3f}, domain_match={int(domain_ok)}, year_hint={int(year_hit)}"
    return float(confidence), rationale


def assist_url_backfill_from_review(cfg: Optional[AssistedBackfillConfig] = None) -> Dict:
    cfg = cfg or AssistedBackfillConfig()

    review_csv = cfg.review_csv or _latest_review_csv()
    if review_csv is None or not Path(review_csv).exists():
        raise FileNotFoundError("No review CSV available. Run --backfill-urls first or provide --assist-backfill-review-csv.")

    if not cfg.canonical_csv.exists():
        raise FileNotFoundError(f"Canonical corpus not found: {cfg.canonical_csv}")

    review_df = pd.read_csv(review_csv)
    canonical_df = pd.read_csv(cfg.canonical_csv)

    if len(review_df) == 0:
        raise ValueError("Review CSV is empty")

    if "proposed_url" not in review_df.columns:
        review_df["proposed_url"] = ""

    review_df["proposed_url"] = review_df["proposed_url"].fillna("").astype(str).str.strip()
    review_df["assisted_proposed_url"] = ""
    review_df["assisted_confidence"] = 0.0
    review_df["assisted_method"] = ""
    review_df["assisted_rationale"] = ""
    review_df["assisted_result_title"] = ""

    pending = review_df[review_df["proposed_url"].eq("")].copy()
    if cfg.max_rows > 0:
        pending = pending.head(int(cfg.max_rows)).copy()

    query_cache: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    assisted_hits = 0

    for idx, row in pending.iterrows():
        title = str(row.get("title", "")).strip()
        source = str(row.get("source", "")).strip()
        date_val = row.get("date", "")
        if not title:
            continue

        cache_key = (_norm_source(source), title)
        if cache_key not in query_cache:
            query_cache[cache_key] = _fetch_search_candidates(
                title=title,
                source=source,
                timeout=int(cfg.request_timeout),
                lookup_url=str(row.get("lookup_url", "") or ""),
            )
        candidates = query_cache[cache_key]

        best_url = ""
        best_title = ""
        best_conf = 0.0
        best_reason = ""

        for cand_url, cand_title in candidates[:20]:
            conf, reason = _score_web_candidate(
                row_title=title,
                row_date=date_val,
                row_source=source,
                cand_url=cand_url,
                cand_title=cand_title,
            )
            if conf > best_conf:
                best_conf = conf
                best_url = cand_url
                best_title = cand_title
                best_reason = reason

        if best_url and best_conf >= float(cfg.min_confidence):
            review_df.at[idx, "assisted_proposed_url"] = best_url
            review_df.at[idx, "assisted_confidence"] = round(float(best_conf), 3)
            review_df.at[idx, "assisted_method"] = "web_search_official_domain"
            review_df.at[idx, "assisted_rationale"] = best_reason
            review_df.at[idx, "assisted_result_title"] = best_title
            assisted_hits += 1

    stamp = _utc_stamp()
    assisted_review_csv = PROCESSED_DATA_DIR / f"url_backfill_assisted_review_{stamp}.csv"
    assisted_patch_csv = PROCESSED_DATA_DIR / f"url_backfill_assisted_patch_{stamp}.csv"
    assisted_preview_csv = PROCESSED_DATA_DIR / f"url_backfill_assisted_preview_canonical_{stamp}.csv"
    assisted_report_json = PROCESSED_DATA_DIR / f"url_backfill_assisted_report_{stamp}.json"

    review_df.to_csv(assisted_review_csv, index=False, encoding="utf-8")

    patch_rows = review_df[review_df["assisted_proposed_url"].fillna("").astype(str).str.strip().ne("")].copy()
    patch_rows.to_csv(assisted_patch_csv, index=False, encoding="utf-8")

    preview_df = canonical_df.copy()
    if len(patch_rows) > 0 and "row_index" in patch_rows.columns and "url" in preview_df.columns:
        for _, r in patch_rows.iterrows():
            try:
                row_idx = int(r.get("row_index"))
                if row_idx in preview_df.index:
                    preview_df.at[row_idx, "url"] = str(r.get("assisted_proposed_url", "")).strip()
            except Exception:
                continue
    preview_df.to_csv(assisted_preview_csv, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_csv": str(cfg.canonical_csv),
        "input_review_csv": str(review_csv),
        "rows_scanned": int(len(pending)),
        "assisted_patch_rows": int(assisted_hits),
        "min_confidence": float(cfg.min_confidence),
        "outputs": {
            "assisted_review_csv": str(assisted_review_csv),
            "assisted_patch_csv": str(assisted_patch_csv),
            "assisted_preview_canonical_csv": str(assisted_preview_csv),
            "assisted_report_json": str(assisted_report_json),
        },
    }

    with open(assisted_report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def _extract_anchor_pairs(html_text: str, base_url: str = "") -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html_text, flags=re.IGNORECASE | re.DOTALL):
        href = html.unescape(str(m.group(1) or "").strip())
        if not href:
            continue
        title_html = str(m.group(2) or "")
        title = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", title_html))).strip()
        if not title:
            continue
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/") and base_url:
            p = urlparse(base_url)
            href = f"{p.scheme}://{p.netloc}{href}"
        if _is_valid_url(href):
            pairs.append((href, title))

    dedup: List[Tuple[str, str]] = []
    seen = set()
    for url, title in pairs:
        key = (url.strip(), _norm_title(title))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((url.strip(), title.strip()))
    return dedup


def _curated_seed_pages_for_source(source: str) -> List[str]:
    src = _norm_source(source)
    seeds = {
        "MEA": [
            "https://www.mea.gov.in/bilateral-documents.htm?dtl/",
            "https://www.mea.gov.in/bilateral-briefs.htm",
            "https://www.mea.gov.in/press-releases.htm",
        ],
        "MOFA": [
            "https://www.mofa.go.jp/region/asia-paci/india/index.html",
            "https://www.mofa.go.jp/announce/index.html",
            "https://www.mofa.go.jp/policy/other/bluebook/index.html",
        ],
        "MOFA_ARCHIVE": [
            "https://www.mofa.go.jp/region/asia-paci/india/index.html",
            "https://www.mofa.go.jp/announce/index.html",
            "https://www.mofa.go.jp/policy/other/bluebook/index.html",
        ],
        "JETRO": [
            "https://www.jetro.go.jp/en/news/",
            "https://www.jetro.go.jp/en/reports/",
        ],
        "EMBJPIN": [
            "https://www.in.emb-japan.go.jp/itprtop_en/index.html",
        ],
    }
    return seeds.get(src, [])


def _collect_curated_archive_candidates(sources: List[str], timeout: int = 20) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    rows: List[Dict] = []
    for src in sorted(set([_norm_source(s) for s in sources if str(s).strip()])):
        for seed in _curated_seed_pages_for_source(src):
            try:
                resp = requests.get(seed, headers=headers, timeout=int(timeout), allow_redirects=True)
            except Exception:
                continue
            if int(resp.status_code) != 200:
                continue

            anchors = _extract_anchor_pairs(resp.text, base_url=str(resp.url))
            for link_url, link_title in anchors:
                rows.append(
                    {
                        "seed_url": seed,
                        "source": src,
                        "candidate_url": link_url,
                        "candidate_title": link_title,
                        "_k_title": _norm_title(link_title),
                        "_k_tokens": _token_set(_norm_title(link_title)),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["seed_url", "source", "candidate_url", "candidate_title", "_k_title", "_k_tokens"])

    out = pd.DataFrame(rows)
    out = out[out["_k_title"].astype(str).str.len() > 0].copy()
    out = out.drop_duplicates(subset=["source", "candidate_url", "_k_title"], keep="first")
    return out


def _score_curated_candidate(row_title: str, row_source: str, candidate_title: str, candidate_url: str) -> Tuple[float, str]:
    target = _norm_title(row_title)
    cand = _norm_title(candidate_title)
    if not target or not cand:
        return 0.0, "missing normalized title"

    sim = SequenceMatcher(None, target, cand).ratio()

    t_tokens = _token_set(target)
    c_tokens = _token_set(cand)
    token_overlap = 0.0
    if t_tokens and c_tokens:
        token_overlap = len(t_tokens.intersection(c_tokens)) / max(1, len(t_tokens.union(c_tokens)))

    host = urlparse(str(candidate_url)).netloc.lower()
    hints = _domain_hints_for_source(row_source)
    domain_ok = any(h in host for h in hints) if hints else True
    domain_score = 1.0 if domain_ok else 0.0

    conf = (0.55 * sim) + (0.30 * token_overlap) + (0.15 * domain_score)
    reason = f"sim={sim:.3f}, token_overlap={token_overlap:.3f}, domain_match={int(domain_ok)}"
    return float(conf), reason


def backfill_from_curated_archives(cfg: Optional[CuratedArchiveBackfillConfig] = None) -> Dict:
    cfg = cfg or CuratedArchiveBackfillConfig()

    review_csv = cfg.review_csv or _latest_review_csv()
    if review_csv is None or not Path(review_csv).exists():
        raise FileNotFoundError("No review CSV available. Run --backfill-urls first or provide --curated-backfill-review-csv.")

    if not cfg.canonical_csv.exists():
        raise FileNotFoundError(f"Canonical corpus not found: {cfg.canonical_csv}")

    review_df = pd.read_csv(review_csv)
    canonical_df = pd.read_csv(cfg.canonical_csv)

    if "proposed_url" not in review_df.columns:
        review_df["proposed_url"] = ""
    review_df["proposed_url"] = review_df["proposed_url"].fillna("").astype(str).str.strip()

    unresolved = review_df[review_df["proposed_url"].eq("")].copy()
    if len(unresolved) == 0:
        raise ValueError("No unresolved rows in review CSV")

    if cfg.max_rows > 0:
        unresolved = unresolved.head(int(cfg.max_rows)).copy()

    curated = _collect_curated_archive_candidates(
        sources=list(unresolved.get("source", pd.Series(dtype=str)).astype(str).tolist()),
        timeout=int(cfg.request_timeout),
    )

    if "curated_proposed_url" not in review_df.columns:
        review_df["curated_proposed_url"] = ""
    review_df["curated_best_url"] = ""
    review_df["curated_best_confidence"] = 0.0
    review_df["curated_best_rationale"] = ""
    review_df["curated_best_title"] = ""
    review_df["curated_confidence"] = 0.0
    review_df["curated_method"] = ""
    review_df["curated_rationale"] = ""
    review_df["curated_result_title"] = ""

    curated_hits = 0
    for idx, row in unresolved.iterrows():
        source = _norm_source(row.get("source", ""))
        title = str(row.get("title", "")).strip()
        if not source or not title:
            continue

        candidates = curated[curated["source"] == source].copy() if len(curated) else pd.DataFrame()
        if len(candidates) == 0:
            continue

        best_conf = 0.0
        best_url = ""
        best_title = ""
        best_reason = ""
        second_conf = 0.0

        for _, cand in candidates.iterrows():
            conf, reason = _score_curated_candidate(
                row_title=title,
                row_source=source,
                candidate_title=str(cand.get("candidate_title", "")),
                candidate_url=str(cand.get("candidate_url", "")),
            )
            if conf > best_conf:
                second_conf = best_conf
                best_conf = conf
                best_url = str(cand.get("candidate_url", "")).strip()
                best_title = str(cand.get("candidate_title", "")).strip()
                best_reason = reason
            elif conf > second_conf:
                second_conf = conf

        margin = best_conf - second_conf
        if best_url:
            review_df.at[idx, "curated_best_url"] = best_url
            review_df.at[idx, "curated_best_confidence"] = round(float(best_conf), 3)
            review_df.at[idx, "curated_best_rationale"] = f"{best_reason}; margin={margin:.3f}"
            review_df.at[idx, "curated_best_title"] = best_title

        if best_url and best_conf >= float(cfg.min_confidence) and margin >= 0.03:
            review_df.at[idx, "curated_proposed_url"] = best_url
            review_df.at[idx, "curated_confidence"] = round(float(best_conf), 3)
            review_df.at[idx, "curated_method"] = "curated_archive_deterministic_match"
            review_df.at[idx, "curated_rationale"] = f"{best_reason}; margin={margin:.3f}"
            review_df.at[idx, "curated_result_title"] = best_title
            curated_hits += 1

    stamp = _utc_stamp()
    curated_review_csv = PROCESSED_DATA_DIR / f"url_backfill_curated_review_{stamp}.csv"
    curated_patch_csv = PROCESSED_DATA_DIR / f"url_backfill_curated_patch_{stamp}.csv"
    curated_preview_csv = PROCESSED_DATA_DIR / f"url_backfill_curated_preview_canonical_{stamp}.csv"
    curated_report_json = PROCESSED_DATA_DIR / f"url_backfill_curated_report_{stamp}.json"

    review_df.to_csv(curated_review_csv, index=False, encoding="utf-8")

    patch_rows = review_df[review_df["curated_proposed_url"].fillna("").astype(str).str.strip().ne("")].copy()
    patch_rows.to_csv(curated_patch_csv, index=False, encoding="utf-8")

    preview_df = canonical_df.copy()
    if len(patch_rows) > 0 and "row_index" in patch_rows.columns and "url" in preview_df.columns:
        for _, r in patch_rows.iterrows():
            try:
                row_idx = int(r.get("row_index"))
                if row_idx in preview_df.index:
                    preview_df.at[row_idx, "url"] = str(r.get("curated_proposed_url", "")).strip()
            except Exception:
                continue
    preview_df.to_csv(curated_preview_csv, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_csv": str(cfg.canonical_csv),
        "input_review_csv": str(review_csv),
        "rows_scanned": int(len(unresolved)),
        "curated_candidate_rows": int(len(curated)),
        "curated_patch_rows": int(curated_hits),
        "min_confidence": float(cfg.min_confidence),
        "outputs": {
            "curated_review_csv": str(curated_review_csv),
            "curated_patch_csv": str(curated_patch_csv),
            "curated_preview_canonical_csv": str(curated_preview_csv),
            "curated_report_json": str(curated_report_json),
        },
    }

    with open(curated_report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def _latest_curated_review_csv() -> Optional[Path]:
    files = sorted(PROCESSED_DATA_DIR.glob("url_backfill_curated_review_*.csv"), reverse=True)
    return files[0] if files else None


def build_semi_auto_approval_csv(cfg: Optional[SemiAutoApprovalConfig] = None) -> Dict:
    cfg = cfg or SemiAutoApprovalConfig()

    curated_review_csv = cfg.curated_review_csv or _latest_curated_review_csv()
    if curated_review_csv is None or not Path(curated_review_csv).exists():
        raise FileNotFoundError("No curated review CSV found. Run --curated-backfill-urls first or pass --semi-auto-curated-review-csv.")

    review_df = pd.read_csv(curated_review_csv)
    if len(review_df) == 0:
        raise ValueError("Curated review CSV is empty")

    for col, default in [
        ("curated_best_url", ""),
        ("curated_best_confidence", 0.0),
        ("curated_best_title", ""),
        ("curated_best_rationale", ""),
    ]:
        if col not in review_df.columns:
            review_df[col] = default

    candidates = review_df[
        review_df["curated_best_url"].fillna("").astype(str).str.strip().ne("")
        & (pd.to_numeric(review_df["curated_best_confidence"], errors="coerce").fillna(0.0) >= float(cfg.medium_min_confidence))
    ].copy()

    approval_rows: List[Dict] = []
    for _, r in candidates.iterrows():
        conf = float(pd.to_numeric(r.get("curated_best_confidence", 0.0), errors="coerce") or 0.0)
        suggested = "APPROVE" if conf >= float(cfg.recommend_approve_confidence) else "REJECT"
        approval_rows.append(
            {
                "row_index": int(r.get("row_index")),
                "source": r.get("source", ""),
                "date": r.get("date", ""),
                "title": r.get("title", ""),
                "current_url": r.get("current_url", ""),
                "candidate_url": r.get("curated_best_url", ""),
                "candidate_title": r.get("curated_best_title", ""),
                "candidate_confidence": round(conf, 3),
                "candidate_rationale": r.get("curated_best_rationale", ""),
                "decision": suggested,
                "review_notes": "",
            }
        )

    stamp = _utc_stamp()
    approval_csv = PROCESSED_DATA_DIR / f"url_backfill_semi_auto_approval_{stamp}.csv"
    approval_report_json = PROCESSED_DATA_DIR / f"url_backfill_semi_auto_report_{stamp}.json"

    approval_df = pd.DataFrame(approval_rows)
    approval_df.to_csv(approval_csv, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_curated_review_csv": str(curated_review_csv),
        "medium_min_confidence": float(cfg.medium_min_confidence),
        "recommend_approve_confidence": float(cfg.recommend_approve_confidence),
        "approval_rows": int(len(approval_df)),
        "outputs": {
            "approval_csv": str(approval_csv),
            "report_json": str(approval_report_json),
        },
    }

    with open(approval_report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def apply_semi_auto_approved_patch(approval_csv: str | Path, canonical_csv: Optional[Path] = None) -> Dict:
    approval_path = Path(approval_csv)
    if not approval_path.exists():
        raise FileNotFoundError(f"Approval CSV not found: {approval_path}")

    canonical_path = canonical_csv or (RAW_DATA_DIR / "india_japan_documents_canonical.csv")
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical CSV not found: {canonical_path}")

    approval_df = pd.read_csv(approval_path)
    canonical_df = pd.read_csv(canonical_path)

    needed_cols = {"row_index", "candidate_url", "decision"}
    if not needed_cols.issubset(set(approval_df.columns)):
        raise ValueError(f"Approval CSV must include columns: {sorted(needed_cols)}")

    approve = approval_df[approval_df["decision"].fillna("").astype(str).str.upper().str.strip() == "APPROVE"].copy()

    applied_rows: List[Dict] = []
    preview_df = canonical_df.copy()
    if "url" not in preview_df.columns:
        preview_df["url"] = ""

    for _, r in approve.iterrows():
        try:
            idx = int(r.get("row_index"))
        except Exception:
            continue
        if idx not in preview_df.index:
            continue

        new_url = str(r.get("candidate_url", "")).strip()
        if not _is_valid_url(new_url):
            continue

        old_url = str(preview_df.at[idx, "url"] if "url" in preview_df.columns else "").strip()
        if old_url == new_url:
            continue

        preview_df.at[idx, "url"] = new_url
        applied_rows.append(
            {
                "row_index": idx,
                "old_url": old_url,
                "new_url": new_url,
                "source": r.get("source", ""),
                "date": r.get("date", ""),
                "title": r.get("title", ""),
            }
        )

    stamp = _utc_stamp()
    final_patch_csv = PROCESSED_DATA_DIR / f"url_backfill_final_patch_{stamp}.csv"
    final_preview_csv = PROCESSED_DATA_DIR / f"url_backfill_final_preview_canonical_{stamp}.csv"
    final_report_json = PROCESSED_DATA_DIR / f"url_backfill_final_patch_report_{stamp}.json"

    pd.DataFrame(applied_rows).to_csv(final_patch_csv, index=False, encoding="utf-8")
    preview_df.to_csv(final_preview_csv, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_approval_csv": str(approval_path),
        "canonical_csv": str(canonical_path),
        "approved_rows": int(len(approve)),
        "applied_patch_rows": int(len(applied_rows)),
        "outputs": {
            "final_patch_csv": str(final_patch_csv),
            "final_preview_canonical_csv": str(final_preview_csv),
            "report_json": str(final_report_json),
        },
    }

    with open(final_report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


if __name__ == "__main__":
    out = generate_url_backfill_patch()
    print(json.dumps(out, indent=2))
