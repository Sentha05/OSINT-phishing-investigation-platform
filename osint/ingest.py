"""Stage 1 - Data acquisition.

The prototype ingests three CSV files (entities, relationships, events).
In a production system this module is where API collectors (RSS, CERT feeds,
MISP/STIX, social-media APIs, WHOIS/passive-DNS) would plug in.  Every record
keeps its `source`, so provenance is never lost downstream.
"""
from pathlib import Path
import pandas as pd

REQUIRED = {
    "entities": {"entity_id", "name", "type"},
    "relationships": {"src", "dst", "rel", "confidence"},
    "events": {"event_id", "timestamp", "source", "text"},
}


def _check(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    missing = REQUIRED[kind] - set(df.columns)
    if missing:
        raise ValueError(f"{kind}.csv is missing required columns: {sorted(missing)}")
    return df


def load_folder(folder):
    folder = Path(folder)
    return (
        _check(pd.read_csv(folder / "entities.csv"), "entities"),
        _check(pd.read_csv(folder / "relationships.csv"), "relationships"),
        _check(pd.read_csv(folder / "events.csv"), "events"),
    )


def load_uploads(entities_file, rels_file, events_file):
    return (
        _check(pd.read_csv(entities_file), "entities"),
        _check(pd.read_csv(rels_file), "relationships"),
        _check(pd.read_csv(events_file), "events"),
    )
