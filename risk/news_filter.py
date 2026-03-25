"""
risk/news_filter.py — High-Impact News Event Filter

Prevents trading around scheduled macro events that cause extreme NQ volatility:
  FOMC decisions, Fed Chair press conferences, CPI/PPI, NFP, GDP, PCE.

Blackout windows (configurable, defaults per spec):
  - 30 minutes BEFORE the event
  - 60 minutes AFTER the event

During a blackout:
  - No new entries
  - Existing positions may continue running (exit manager handles stop/target)
  - The bot logs a warning and skips the trade

Two event sources supported:
  1. Static schedule — hard-coded recurring events (FOMC, NFP, CPI by typical
     release day/time). Good enough for backtesting and a first live pass.
  2. Dynamic schedule — caller provides a list of NewsEvent objects loaded
     from an external calendar (e.g. Economic Calendar API, Benzinga, Quandl).
     NewsFilter merges both sources and deduplicates.

Usage:
    from risk.news_filter import NewsFilter, NewsEvent
    import pandas as pd

    # Static-only (backtesting):
    nf = NewsFilter(config)
    if nf.is_blackout(pd.Timestamp("2025-10-29 14:00", tz="UTC")):
        return  # skip — near FOMC

    # With dynamic events (live):
    events = [
        NewsEvent("CPI Release", pd.Timestamp("2025-11-13 13:30", tz="UTC"), impact="high"),
        NewsEvent("FOMC Decision", pd.Timestamp("2025-11-07 19:00", tz="UTC"), impact="high"),
    ]
    nf.load_events(events)
    check = nf.check(pd.Timestamp.now(tz="UTC"))
    if not check.can_trade:
        logger.warning(check.reason)
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from loguru import logger

ImpactLevel = Literal["high", "medium", "low"]

# Default blackout windows in minutes
_DEFAULT_MINUTES_BEFORE = 30
_DEFAULT_MINUTES_AFTER  = 60

# Static recurring events — (description, weekday_name, hour_et, minute_et)
# These repeat on their typical schedule; the filter checks within a rolling
# 90-day window generated at runtime.
_STATIC_RECURRING: list[tuple[str, str, int, int]] = [
    # NFP — first Friday of each month at 8:30 ET
    ("NFP",         "first_friday", 8, 30),
    # CPI — second Tuesday of each month at 8:30 ET (approximate)
    ("CPI",         "second_tuesday", 8, 30),
    # PCE — last Friday of each month at 8:30 ET (approximate)
    ("PCE",         "last_friday", 8, 30),
    # GDP — last Wednesday of each month at 8:30 ET (approximate)
    ("GDP",         "last_wednesday", 8, 30),
]

# FOMC dates are irregular — these are the 2025 FOMC decision dates/times (ET)
# Format: (year, month, day, hour, minute)
_FOMC_DATES: list[tuple[int, int, int, int, int]] = [
    (2025,  1, 29, 14,  0),
    (2025,  3, 19, 14,  0),
    (2025,  5,  7, 14,  0),
    (2025,  6, 18, 14,  0),
    (2025,  7, 30, 14,  0),
    (2025,  9, 17, 14,  0),
    (2025, 10, 29, 14,  0),
    (2025, 12, 10, 14,  0),
    (2026,  1, 28, 14,  0),
    (2026,  3, 18, 14,  0),
    (2026,  5,  6, 14,  0),
    (2026,  6, 17, 14,  0),
    (2026,  7, 29, 14,  0),
    (2026,  9, 16, 14,  0),
    (2026, 10, 28, 14,  0),
    (2026, 12,  9, 14,  0),
]

_ET = "America/New_York"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NewsEvent:
    """A single scheduled macro event."""
    name:       str
    timestamp:  pd.Timestamp          # UTC
    impact:     ImpactLevel = "high"
    source:     str         = "static"

    def __post_init__(self):
        # Ensure UTC
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.tz_localize("UTC")
        elif str(self.timestamp.tzinfo) != "UTC":
            self.timestamp = self.timestamp.tz_convert("UTC")

    def __str__(self) -> str:
        return (
            f"NewsEvent({self.name} | {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"impact={self.impact})"
        )


@dataclass
class NewsFilterCheck:
    """Output of NewsFilter.check()."""
    can_trade:    bool
    reason:       str
    active_event: NewsEvent | None = None

    def __str__(self) -> str:
        status = "OK" if self.can_trade else f"BLOCKED({self.reason})"
        event_str = str(self.active_event) if self.active_event else "none"
        return f"NewsFilterCheck({status} | event={event_str})"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NewsFilter:
    """
    Blocks trading during scheduled high-impact news event windows.

    Args:
        config:         top-level Config dataclass. If None, uses spec defaults.
        minutes_before: Minutes before event to start blackout.
        minutes_after:  Minutes after event to end blackout.
    """

    def __init__(
        self,
        config         = None,
        minutes_before: int | None = None,
        minutes_after:  int | None = None,
    ):
        if config is not None:
            self._minutes_before = getattr(
                config, "news_blackout_minutes_before", _DEFAULT_MINUTES_BEFORE
            )
            self._minutes_after = getattr(
                config, "news_blackout_minutes_after", _DEFAULT_MINUTES_AFTER
            )
        else:
            self._minutes_before = _DEFAULT_MINUTES_BEFORE
            self._minutes_after  = _DEFAULT_MINUTES_AFTER

        # Allow direct overrides
        if minutes_before is not None:
            self._minutes_before = minutes_before
        if minutes_after is not None:
            self._minutes_after = minutes_after

        # Dynamic event list (from external calendar)
        self._dynamic_events: list[NewsEvent] = []

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def check(self, now: pd.Timestamp) -> NewsFilterCheck:
        """
        Check whether trading is allowed at `now`.

        Args:
            now: Current UTC timestamp.

        Returns:
            NewsFilterCheck. can_trade=False if inside a blackout window.
        """
        now_utc = _ensure_utc(now)

        active = self._find_active_event(now_utc)
        if active is None:
            return NewsFilterCheck(can_trade=True, reason="ok", active_event=None)

        minutes_from_event = (now_utc - active.timestamp).total_seconds() / 60
        if minutes_from_event < 0:
            window = f"{abs(minutes_from_event):.0f}m before {active.name}"
        else:
            window = f"{minutes_from_event:.0f}m after {active.name}"

        result = NewsFilterCheck(
            can_trade    = False,
            reason       = f"news_blackout: {window}",
            active_event = active,
        )
        logger.warning(f"[news] {result}")
        return result

    def is_blackout(self, now: pd.Timestamp) -> bool:
        """Convenience wrapper — returns True if inside a blackout window."""
        return not self.check(now).can_trade

    def load_events(self, events: list[NewsEvent]) -> None:
        """
        Load dynamic events from an external source (e.g. economic calendar API).

        Merges with existing dynamic events. Duplicate timestamps + names are
        deduplicated. Static FOMC/NFP/CPI events are unaffected.

        Args:
            events: List of NewsEvent objects (high impact only recommended).
        """
        existing_keys = {(e.name, e.timestamp) for e in self._dynamic_events}
        added = 0
        for ev in events:
            ev_utc = NewsEvent(ev.name, ev.timestamp, ev.impact, ev.source)
            key = (ev_utc.name, ev_utc.timestamp)
            if key not in existing_keys:
                self._dynamic_events.append(ev_utc)
                existing_keys.add(key)
                added += 1
        logger.debug(f"[news] Loaded {added} new dynamic events ({len(self._dynamic_events)} total)")

    def clear_dynamic_events(self) -> None:
        """Remove all dynamically loaded events (static schedule is preserved)."""
        self._dynamic_events.clear()

    def upcoming_events(
        self,
        now:    pd.Timestamp,
        hours:  int = 24,
    ) -> list[NewsEvent]:
        """
        Return all events in the next `hours` hours from `now`.

        Args:
            now:   Current UTC timestamp.
            hours: Look-ahead window in hours (default 24).

        Returns:
            List of NewsEvent sorted by timestamp.
        """
        now_utc  = _ensure_utc(now)
        end_utc  = now_utc + pd.Timedelta(hours=hours)
        all_evs  = self._get_all_events(now_utc)
        upcoming = [e for e in all_evs if now_utc <= e.timestamp <= end_utc]
        return sorted(upcoming, key=lambda e: e.timestamp)

    def next_event(self, now: pd.Timestamp) -> NewsEvent | None:
        """Return the next upcoming high-impact event, or None."""
        events = self.upcoming_events(now, hours=7 * 24)
        return events[0] if events else None

    def minutes_to_next_event(self, now: pd.Timestamp) -> float | None:
        """Return minutes until the next event, or None if no event in next 7 days."""
        ev = self.next_event(now)
        if ev is None:
            return None
        delta = (ev.timestamp - _ensure_utc(now)).total_seconds() / 60
        return max(0.0, delta)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_active_event(self, now_utc: pd.Timestamp) -> NewsEvent | None:
        """Return the first event whose blackout window contains now_utc."""
        all_events = self._get_all_events(now_utc)
        before_td  = pd.Timedelta(minutes=self._minutes_before)
        after_td   = pd.Timedelta(minutes=self._minutes_after)

        for ev in all_events:
            window_start = ev.timestamp - before_td
            window_end   = ev.timestamp + after_td
            if window_start <= now_utc <= window_end:
                return ev
        return None

    def _get_all_events(self, reference: pd.Timestamp) -> list[NewsEvent]:
        """Return all events (static + dynamic) within ±90 days of reference."""
        events = list(self._dynamic_events)
        events.extend(_build_static_events(reference))
        return events


# ---------------------------------------------------------------------------
# Static event generation
# ---------------------------------------------------------------------------

def _build_static_events(reference: pd.Timestamp) -> list[NewsEvent]:
    """
    Generate static recurring events within a ±90-day window of reference.

    FOMC dates are from the hard-coded list.
    NFP/CPI/PCE/GDP are approximated from recurring day patterns.
    """
    events: list[NewsEvent] = []
    window_start = reference - pd.Timedelta(days=90)
    window_end   = reference + pd.Timedelta(days=90)

    # FOMC dates
    for year, month, day, hour, minute in _FOMC_DATES:
        try:
            ts_et  = pd.Timestamp(year, month, day, hour, minute,
                                  tz=_ET)
            ts_utc = ts_et.tz_convert("UTC")
            if window_start <= ts_utc <= window_end:
                events.append(NewsEvent("FOMC", ts_utc, "high", "static"))
        except Exception:
            pass

    # Recurring events
    months = _months_in_window(window_start, window_end)
    for year, month in months:
        for name, day_spec, hour_et, min_et in _STATIC_RECURRING:
            day = _resolve_day(year, month, day_spec)
            if day is None:
                continue
            try:
                ts_et  = pd.Timestamp(year, month, day, hour_et, min_et, tz=_ET)
                ts_utc = ts_et.tz_convert("UTC")
                if window_start <= ts_utc <= window_end:
                    events.append(NewsEvent(name, ts_utc, "high", "static"))
            except Exception:
                pass

    return events


def _months_in_window(
    start: pd.Timestamp,
    end:   pd.Timestamp,
) -> list[tuple[int, int]]:
    """Return (year, month) pairs for every month overlapping [start, end]."""
    result = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        result.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result


def _resolve_day(year: int, month: int, day_spec: str) -> int | None:
    """
    Resolve a day_spec string to a calendar day number.

    Supported specs:
      "first_friday", "second_tuesday", "last_friday", "last_wednesday"
    """
    _WEEKDAY = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }

    parts = day_spec.split("_")
    if len(parts) != 2:
        return None

    ordinal, weekday_name = parts[0], parts[1]
    target_weekday = _WEEKDAY.get(weekday_name)
    if target_weekday is None:
        return None

    cal = calendar.monthcalendar(year, month)
    # Each row is Mon–Sun; collect all days matching target_weekday
    days = [week[target_weekday] for week in cal if week[target_weekday] != 0]

    if ordinal == "first" and len(days) >= 1:
        return days[0]
    elif ordinal == "second" and len(days) >= 2:
        return days[1]
    elif ordinal == "last" and len(days) >= 1:
        return days[-1]
    return None


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return ts with UTC timezone attached."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
