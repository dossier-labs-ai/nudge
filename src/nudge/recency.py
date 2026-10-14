"""Decision-time weighting shared by fitting, selection, and discovery."""

from datetime import datetime, timezone

from .types import finite

DEFAULT_HALF_LIFE_SECONDS = 7 * 24 * 60 * 60


def utc_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def validate_half_life(half_life_seconds: float | None) -> None:
    if half_life_seconds is not None and finite(half_life_seconds, "half_life_seconds") <= 0:
        raise ValueError("half_life_seconds must be positive, or None for unit weights")


def weight_rows(rows: list[dict], half_life_seconds: float | None) -> tuple[list[dict], str | None]:
    validate_half_life(half_life_seconds)
    if not rows:
        return [], None
    # Normalize timestamps rather than sorting ISO strings with mixed UTC suffixes.
    dated = [(datetime.fromisoformat(row["created_at"]), row) for row in rows]
    for date, _ in dated:
        utc_timestamp(date)
    dated.sort(key=lambda pair: pair[0], reverse=True)
    reference = dated[0][0]
    return [
        {
            **row,
            "weight": 1.0
            if half_life_seconds is None
            else 2.0 ** (-(reference - date).total_seconds() / half_life_seconds),
        }
        for date, row in dated
    ], utc_timestamp(reference)
