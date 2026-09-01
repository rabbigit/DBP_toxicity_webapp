"""Privacy-conscious usage analytics for the Streamlit application.

Analytics are optional. Every public function fails closed so an unavailable
analytics service can never prevent a toxicity prediction.
"""

from __future__ import annotations

import hashlib
import hmac
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import streamlit as st


VALID_EVENTS = {"visit", "single_prediction", "batch_prediction"}


def _settings() -> dict[str, str]:
    """Return analytics secrets, or an empty mapping when not configured."""
    try:
        section = st.secrets.get("analytics", {})
        return {
            "supabase_url": str(section.get("supabase_url", "")).rstrip("/"),
            "supabase_key": str(section.get("supabase_key", "")),
            "visitor_salt": str(section.get("visitor_salt", "")),
            "dashboard_password": str(section.get("dashboard_password", "")),
        }
    except Exception:
        return {}


def analytics_enabled() -> bool:
    settings = _settings()
    return all(
        settings.get(key)
        for key in ("supabase_url", "supabase_key", "visitor_salt")
    )


def dashboard_password() -> str:
    return _settings().get("dashboard_password", "")


def _headers(prefer: str | None = None) -> dict[str, str]:
    settings = _settings()
    headers = {
        "apikey": settings["supabase_key"],
        "Content-Type": "application/json",
    }
    # New sb_secret_ keys must only use the apikey header. Legacy
    # service_role JWTs also require Authorization: Bearer.
    if not settings["supabase_key"].startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {settings['supabase_key']}"
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _session_id() -> str:
    if "analytics_session_id" not in st.session_state:
        st.session_state.analytics_session_id = str(uuid.uuid4())
    return st.session_state.analytics_session_id


def _visitor_hash(ip_address: str | None) -> str:
    """Create a pseudonymous, approximate visitor identifier.

    Combining IP and user agent distinguishes most browsers without storing
    either value. It remains an estimate because IP addresses can change or be
    shared. A random per-session fallback is used in local development.
    """
    settings = _settings()
    user_agent = str(st.context.headers.get("user-agent", "unknown"))
    source = f"{ip_address or _session_id()}|{user_agent}".encode("utf-8")
    return hmac.new(
        settings["visitor_salt"].encode("utf-8"),
        source,
        hashlib.sha256,
    ).hexdigest()


def _coarse_location(ip_address: str | None) -> dict[str, Any]:
    """Resolve an IP to coarse geography and never return the original IP."""
    empty = {
        "country_code": None,
        "country_name": None,
        "region": None,
        "latitude": None,
        "longitude": None,
    }
    if not ip_address:
        return empty

    try:
        response = requests.get(
            f"https://ipapi.co/{ip_address}/json/",
            timeout=4,
            headers={"User-Agent": "DBP-Toxicity-Predictor/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            return empty

        latitude = payload.get("latitude")
        longitude = payload.get("longitude")
        return {
            "country_code": payload.get("country_code"),
            "country_name": payload.get("country_name"),
            "region": payload.get("region"),
            # Whole-degree coordinates provide only a general map location.
            "latitude": round(float(latitude)) if latitude is not None else None,
            "longitude": round(float(longitude)) if longitude is not None else None,
        }
    except (requests.RequestException, TypeError, ValueError):
        return empty


def _session_location(ip_address: str | None) -> dict[str, Any]:
    """Resolve geography no more than once during a browser session."""
    if "analytics_coarse_location" not in st.session_state:
        st.session_state.analytics_coarse_location = _coarse_location(ip_address)
    return st.session_state.analytics_coarse_location


def _insert_event(event_type: str, item_count: int) -> bool:
    if not analytics_enabled() or event_type not in VALID_EVENTS:
        return False

    try:
        ip_address = st.context.ip_address
        payload = {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "visitor_hash": _visitor_hash(ip_address),
            "session_id": _session_id(),
            "event_type": event_type,
            "item_count": max(int(item_count), 0),
            **_session_location(ip_address),
        }
        response = requests.post(
            f"{_settings()['supabase_url']}/rest/v1/analytics_events",
            headers=_headers("return=minimal"),
            json=payload,
            timeout=5,
        )
        response.raise_for_status()
        return True
    except Exception:
        # Usage tracking must never interrupt the scientific application.
        return False


def log_visit_once() -> None:
    """Record at most one visit per Streamlit browser session."""
    if st.session_state.get("analytics_visit_attempted"):
        return
    st.session_state.analytics_visit_attempted = True
    _insert_event("visit", item_count=0)


def log_prediction(event_type: str, item_count: int) -> None:
    """Record a completed single or batch prediction."""
    _insert_event(event_type, item_count=item_count)


def fetch_events() -> pd.DataFrame:
    """Read analytics events for the private dashboard."""
    columns = [
        "event_time",
        "visitor_hash",
        "session_id",
        "event_type",
        "item_count",
        "country_code",
        "country_name",
        "region",
        "latitude",
        "longitude",
    ]
    if not analytics_enabled():
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    page_size = 1000
    try:
        for start in range(0, 100_000, page_size):
            headers = _headers()
            headers["Range"] = f"{start}-{start + page_size - 1}"
            response = requests.get(
                f"{_settings()['supabase_url']}/rest/v1/analytics_events",
                headers=headers,
                params={"select": ",".join(columns), "order": "event_time.asc"},
                timeout=10,
            )
            response.raise_for_status()
            page = response.json()
            records.extend(page)
            if len(page) < page_size:
                break
    except Exception as error:
        raise RuntimeError("The analytics database could not be read.") from error

    frame = pd.DataFrame(records, columns=columns)
    if not frame.empty:
        frame["event_time"] = pd.to_datetime(
            frame["event_time"], utc=True, errors="coerce"
        )
        frame["item_count"] = pd.to_numeric(
            frame["item_count"], errors="coerce"
        ).fillna(0).astype(int)
    return frame
