"""
tests/test_news_filter.py — Unit tests for NewsFilter
"""

import pytest
import pandas as pd
from risk.news_filter import (
    NewsFilter, NewsEvent, NewsFilterCheck,
    _resolve_day, _build_static_events, _ensure_utc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts(dt_str: str, tz: str = "UTC") -> pd.Timestamp:
    return pd.Timestamp(dt_str, tz=tz)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nf():
    return NewsFilter()


@pytest.fixture
def fomc_event():
    """A synthetic FOMC event on a date NOT in the static list (2027-04-15 18:00 UTC)."""
    return NewsEvent("FOMC", ts("2027-04-15 18:00"))


# ---------------------------------------------------------------------------
# 1. NewsEvent normalization
# ---------------------------------------------------------------------------

class TestNewsEvent:

    def test_utc_passthrough(self):
        ev = NewsEvent("CPI", ts("2025-11-13 13:30"))
        assert str(ev.timestamp.tzinfo) == "UTC"

    def test_et_converted_to_utc(self):
        ev = NewsEvent("NFP", pd.Timestamp("2025-11-07 08:30", tz="America/New_York"))
        assert str(ev.timestamp.tzinfo) == "UTC"
        # 8:30 ET in November (EST = UTC-5) → 13:30 UTC
        assert ev.timestamp.hour == 13
        assert ev.timestamp.minute == 30

    def test_naive_localized_to_utc(self):
        ev = NewsEvent("GDP", pd.Timestamp("2025-11-26 13:30"))
        assert str(ev.timestamp.tzinfo) == "UTC"

    def test_str_no_error(self, fomc_event):
        s = str(fomc_event)
        assert "FOMC" in s


# ---------------------------------------------------------------------------
# 2. is_blackout / check — core window logic
# ---------------------------------------------------------------------------

class TestBlackoutWindow:

    def test_well_before_event_ok(self, nf, fomc_event):
        nf.load_events([fomc_event])
        # 2 hours before → outside 30m window
        t = fomc_event.timestamp - pd.Timedelta(minutes=120)
        assert nf.is_blackout(t) is False

    def test_just_before_blackout_ok(self, nf, fomc_event):
        nf.load_events([fomc_event])
        # 31 minutes before → just outside window
        t = fomc_event.timestamp - pd.Timedelta(minutes=31)
        assert nf.is_blackout(t) is False

    def test_inside_pre_event_window_blocked(self, nf, fomc_event):
        nf.load_events([fomc_event])
        # 15 minutes before → inside 30m pre-event window
        t = fomc_event.timestamp - pd.Timedelta(minutes=15)
        assert nf.is_blackout(t) is True

    def test_at_event_time_blocked(self, nf, fomc_event):
        nf.load_events([fomc_event])
        assert nf.is_blackout(fomc_event.timestamp) is True

    def test_inside_post_event_window_blocked(self, nf, fomc_event):
        nf.load_events([fomc_event])
        # 45 minutes after → inside 60m post-event window
        t = fomc_event.timestamp + pd.Timedelta(minutes=45)
        assert nf.is_blackout(t) is True

    def test_after_post_event_window_ok(self, nf, fomc_event):
        nf.load_events([fomc_event])
        # 61 minutes after → outside window
        t = fomc_event.timestamp + pd.Timedelta(minutes=61)
        assert nf.is_blackout(t) is False

    def test_check_returns_event_when_blocked(self, nf, fomc_event):
        nf.load_events([fomc_event])
        t = fomc_event.timestamp - pd.Timedelta(minutes=10)
        result = nf.check(t)
        assert result.can_trade is False
        assert result.active_event is not None
        assert "FOMC" in result.active_event.name

    def test_check_active_event_none_when_ok(self, nf, fomc_event):
        nf.load_events([fomc_event])
        t = fomc_event.timestamp + pd.Timedelta(minutes=120)
        result = nf.check(t)
        assert result.can_trade is True
        assert result.active_event is None


# ---------------------------------------------------------------------------
# 3. Custom window sizes
# ---------------------------------------------------------------------------

class TestCustomWindows:

    def test_custom_minutes_before(self, fomc_event):
        nf = NewsFilter(minutes_before=10, minutes_after=20)
        nf.load_events([fomc_event])
        # 15 min before → outside custom 10m window
        assert nf.is_blackout(fomc_event.timestamp - pd.Timedelta(minutes=15)) is False
        # 5 min before → inside custom 10m window
        assert nf.is_blackout(fomc_event.timestamp - pd.Timedelta(minutes=5)) is True

    def test_custom_minutes_after(self, fomc_event):
        nf = NewsFilter(minutes_before=5, minutes_after=15)
        nf.load_events([fomc_event])
        # 20 min after → outside custom 15m window
        assert nf.is_blackout(fomc_event.timestamp + pd.Timedelta(minutes=20)) is False
        # 10 min after → inside custom 15m window
        assert nf.is_blackout(fomc_event.timestamp + pd.Timedelta(minutes=10)) is True


# ---------------------------------------------------------------------------
# 4. Dynamic events
# ---------------------------------------------------------------------------

class TestDynamicEvents:

    def test_load_events(self, nf):
        ev = NewsEvent("PPI", ts("2026-01-14 13:30"))
        nf.load_events([ev])
        # Inside the window
        assert nf.is_blackout(ts("2026-01-14 13:15")) is True

    def test_duplicate_events_deduplicated(self, nf):
        ev = NewsEvent("CPI", ts("2026-02-12 13:30"))
        nf.load_events([ev])
        nf.load_events([ev])   # load same event twice
        assert len(nf._dynamic_events) == 1

    def test_clear_dynamic_events(self, nf):
        ev = NewsEvent("PPI", ts("2026-01-15 13:30"))
        nf.load_events([ev])
        nf.clear_dynamic_events()
        assert nf._dynamic_events == []
        # Should no longer block
        assert nf.is_blackout(ts("2026-01-15 13:15")) is False


# ---------------------------------------------------------------------------
# 5. Static FOMC events
# ---------------------------------------------------------------------------

class TestStaticFOMC:

    def test_known_fomc_2025_oct_blocked(self, nf):
        # 2025-10-29 14:00 ET is in the static list
        # During EDT (October), ET = UTC-4 → 14:00 ET = 18:00 UTC
        t_utc = ts("2025-10-29 18:00")  # exactly at FOMC time
        assert nf.is_blackout(t_utc) is True

    def test_known_fomc_day_before_ok(self, nf):
        # Day before FOMC, same time → no blackout
        t_utc = ts("2025-10-28 18:00")
        assert nf.is_blackout(t_utc) is False

    def test_known_fomc_2026_jan_blocked(self, nf):
        # 2026-01-28 14:00 ET — January (EST = UTC-5) → 19:00 UTC
        t_utc = ts("2026-01-28 19:00")
        assert nf.is_blackout(t_utc) is True


# ---------------------------------------------------------------------------
# 6. Upcoming events / next event
# ---------------------------------------------------------------------------

class TestUpcomingEvents:

    def test_upcoming_includes_future_events(self, nf):
        ev = NewsEvent("CPI", ts("2026-04-10 13:30"))
        nf.load_events([ev])
        now = ts("2026-04-10 12:00")
        upcoming = nf.upcoming_events(now, hours=4)
        assert any(e.name == "CPI" for e in upcoming)

    def test_upcoming_excludes_past_events(self, nf):
        ev = NewsEvent("CPI", ts("2026-04-10 13:30"))
        nf.load_events([ev])
        now = ts("2026-04-10 15:00")   # after the event
        upcoming = nf.upcoming_events(now, hours=4)
        assert not any(e.name == "CPI" for e in upcoming)

    def test_upcoming_sorted_by_timestamp(self, nf):
        now = ts("2026-01-01 00:00")
        ev1 = NewsEvent("B_event", ts("2026-01-01 15:00"))
        ev2 = NewsEvent("A_event", ts("2026-01-01 13:00"))
        nf.load_events([ev1, ev2])
        upcoming = nf.upcoming_events(now, hours=24)
        ts_list = [e.timestamp for e in upcoming]
        assert ts_list == sorted(ts_list)

    def test_minutes_to_next_event(self, nf):
        ev = NewsEvent("NFP", ts("2026-03-06 13:30"))
        nf.load_events([ev])
        now = ts("2026-03-06 13:00")
        mins = nf.minutes_to_next_event(now)
        assert mins == pytest.approx(30.0)

    def test_minutes_to_next_event_none_if_empty(self):
        nf = NewsFilter()
        # Use a date far from all static events
        now = ts("2030-06-15 12:00")
        # No static events in 2030 — should return None
        result = nf.minutes_to_next_event(now)
        assert result is None


# ---------------------------------------------------------------------------
# 7. _resolve_day helper
# ---------------------------------------------------------------------------

class TestResolveDay:

    def test_first_friday_jan_2025(self):
        # Jan 2025: first Friday = 3rd
        assert _resolve_day(2025, 1, "first_friday") == 3

    def test_second_tuesday_jan_2025(self):
        # Jan 2025 Tuesdays: 7, 14, 21, 28 → second = 14
        assert _resolve_day(2025, 1, "second_tuesday") == 14

    def test_last_friday_jan_2025(self):
        # Jan 2025 Fridays: 3, 10, 17, 24, 31 → last = 31
        assert _resolve_day(2025, 1, "last_friday") == 31

    def test_last_wednesday_jan_2025(self):
        # Jan 2025 Wednesdays: 1, 8, 15, 22, 29 → last = 29
        assert _resolve_day(2025, 1, "last_wednesday") == 29

    def test_invalid_spec_returns_none(self):
        assert _resolve_day(2025, 1, "invalid_spec") is None

    def test_bad_weekday_returns_none(self):
        assert _resolve_day(2025, 1, "first_holiday") is None


# ---------------------------------------------------------------------------
# 8. _ensure_utc helper
# ---------------------------------------------------------------------------

class TestEnsureUtc:

    def test_naive_gets_utc(self):
        t = _ensure_utc(pd.Timestamp("2025-01-01 12:00"))
        assert str(t.tzinfo) == "UTC"

    def test_et_converted(self):
        t_et = pd.Timestamp("2025-01-01 07:00", tz="America/New_York")
        t = _ensure_utc(t_et)
        assert str(t.tzinfo) == "UTC"
        assert t.hour == 12  # EST = UTC-5


# ---------------------------------------------------------------------------
# 9. check() __str__
# ---------------------------------------------------------------------------

class TestCheckStr:

    def test_str_ok(self, nf):
        result = nf.check(ts("2030-01-01 12:00"))
        s = str(result)
        assert "OK" in s

    def test_str_blocked(self, nf, fomc_event):
        nf.load_events([fomc_event])
        result = nf.check(fomc_event.timestamp)
        s = str(result)
        assert "BLOCKED" in s
