"""`current_datetime` must resolve real IANA zones, and say so when it cannot.

`zoneinfo` reads the operating system's IANA database. Linux and macOS ship one;
Windows does not, and Python's fallback is the `tzdata` PyPI package. It was not
declared in `pyproject.toml`, so on a fresh Windows install every zone failed:

    >>> current_datetime("Asia/Kolkata")
    ValueError: unknown IANA timezone: Asia/Kolkata

The key is valid. The message blames the caller's input for a missing dependency,
which is the most expensive kind of wrong error: it sends you looking at the
timezone string instead of the environment. Section 8 asks for schedules in UTC
"converted for display only", so a tool that cannot convert affects every
scheduled summary the agent writes.

Observed on Windows 11 / CPython 3.13 before this change:
`tests/test_general_tools.py::test_current_datetime_uses_iana_timezone` failed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from zoneinfo import ZoneInfoNotFoundError

import pytest

from s16code import tools
from s16code.tools import current_datetime

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_tzdata_is_a_declared_dependency() -> None:
    """The database is a dependency, not something the host is assumed to have."""
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["dependencies"]
    names = [entry.split(">")[0].split("=")[0].split("[")[0].split(";")[0].strip().lower()
             for entry in declared]
    assert "tzdata" in names, (
        "tzdata is not declared, so zoneinfo has no database on Windows and every "
        "IANA key raises there while passing on Linux CI"
    )


@pytest.mark.parametrize("zone", ["Asia/Kolkata", "America/New_York", "Europe/London", "UTC"])
def test_real_zones_resolve(zone: str) -> None:
    result = current_datetime(zone)
    assert result["timezone"] == zone
    assert result["iso"]
    assert result["weekday"]


def test_a_genuinely_unknown_key_is_still_a_value_error() -> None:
    """Distinguishing the two cases must not swallow the ordinary one."""
    with pytest.raises(ValueError, match="unknown IANA timezone"):
        current_datetime("Amrica/New_York")


def test_a_missing_database_is_not_reported_as_a_bad_key(monkeypatch) -> None:
    """The diagnosis must name the cause, not accuse the input.

    Simulates the Windows-without-tzdata interpreter: every key raises, and
    nothing is enumerable. The message must point at the package, not the key.
    """
    def no_database(_key):
        raise ZoneInfoNotFoundError("No time zone found with key")

    monkeypatch.setattr(tools, "ZoneInfo", no_database)
    monkeypatch.setattr(tools, "available_timezones", lambda: set())
    with pytest.raises(RuntimeError, match="tzdata"):
        current_datetime("Asia/Kolkata")
