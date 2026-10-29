"""Stage 2 - Pre-processing.

Normalise timestamps to UTC, re-fang defanged indicators, drop duplicates,
validate coordinates and convert the Admiralty/NATO source grading
(reliability A-F x credibility 1-6) into a 0-1 confidence score.
Every cleaning step is counted so the analyst can see what was changed.
"""
import re
import pandas as pd

_DEFANG = [
    (r"hxxp", "http"),
    (r"\[\.\]|\(\.\)|\[dot\]", "."),
    (r"\[at\]|\(at\)", "@"),
]
REL = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.3}   # F = cannot be judged
CRED = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2, 6: 0.3}               # 6 = cannot be judged


def refang(text: str) -> str:
    for pat, rep in _DEFANG:
        text = re.sub(pat, rep, text, flags=re.I)
    return text


def _cred_score(v):
    try:
        return CRED.get(int(v), 0.3)
    except (TypeError, ValueError):
        return 0.3


def clean_events(events: pd.DataFrame):
    ev = events.copy()
    qa = {"rows_in": int(len(ev))}

    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True, errors="coerce")
    qa["bad_timestamps_dropped"] = int(ev["timestamp"].isna().sum())
    ev = ev.dropna(subset=["timestamp"])

    ev["text_clean"] = ev["text"].fillna("").astype(str).map(refang)
    n = len(ev)
    ev = ev.drop_duplicates(subset=["timestamp", "source", "text_clean"])
    qa["duplicates_removed"] = n - len(ev)

    for c in ("lat", "lon"):
        ev[c] = pd.to_numeric(ev.get(c), errors="coerce")
    ev["has_geo"] = ev["lat"].between(-90, 90) & ev["lon"].between(-180, 180)
    qa["events_without_valid_geo"] = int((~ev["has_geo"]).sum())

    for col, default in (("geo_precision", "none"), ("event_type", "unknown"),
                         ("source_reliability", "F"), ("info_credibility", 6)):
        if col not in ev.columns:
            ev[col] = default
    ev["geo_precision"] = ev["geo_precision"].fillna("none").astype(str)
    ev["event_type"] = ev["event_type"].fillna("unknown").astype(str)

    rel = ev["source_reliability"].astype(str).str.upper().map(REL).fillna(0.3)
    cred = ev["info_credibility"].map(_cred_score)
    ev["confidence"] = (rel * cred).round(3)

    qa["rows_out"] = int(len(ev))
    return ev.reset_index(drop=True), qa
