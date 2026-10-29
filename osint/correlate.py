"""Stage 4 - Correlation.

Builds the node and edge tables.  Every edge carries a `kind` so the analyst can
always tell *reported* facts from *inferred* links:

  reported       - stated in the relationships register by a source
  co-mention     - two entities appear in the same event (INFERRED, not proof)
  observed-with  - an entity and an indicator appear in the same event (INFERRED)

Fuzzy entity merges are only *suggested*; the analyst decides.
"""
import difflib
import itertools
from collections import defaultdict
import pandas as pd

EDGE_COLS = ["src", "dst", "kind", "label", "weight", "confidence",
             "first_seen", "last_seen", "evidence"]


def merge_candidates(entities: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    out = []
    for a, b in itertools.combinations(entities.itertuples(), 2):
        if a.type != b.type:
            continue
        s = difflib.SequenceMatcher(None, str(a.name).lower(), str(b.name).lower()).ratio()
        if s >= threshold:
            out.append({"entity_a": a.entity_id, "name_a": a.name,
                        "entity_b": b.entity_id, "name_b": b.name,
                        "similarity": round(s, 2)})
    return pd.DataFrame(out, columns=["entity_a", "name_a", "entity_b", "name_b", "similarity"])


def _aggregate(pairs: dict, ev: pd.DataFrame, kind: str, label: str) -> list:
    rows = []
    for (a, b), ids in pairs.items():
        sub = ev.loc[ev.index.intersection(ids)]
        rows.append({
            "src": a, "dst": b, "kind": kind, "label": label,
            "weight": len(ids),
            "confidence": round(float(sub["confidence"].mean()), 3) if len(sub) else 0.0,
            "first_seen": sub["timestamp"].min(), "last_seen": sub["timestamp"].max(),
            "evidence": ";".join(sorted(ids)),
        })
    return rows


def build_graph_tables(entities, rels, events, mentions, indicators):
    ev = events.set_index("event_id")

    nodes = entities[["entity_id", "name", "type"]].rename(columns={"entity_id": "node_id"})
    ioc_nodes = (indicators.drop_duplicates("value")
                 .assign(node_id=lambda d: "IOC:" + d["value"],
                         name=lambda d: d["value"],
                         type=lambda d: "indicator:" + d["ioc_type"])
                 [["node_id", "name", "type"]])
    nodes = pd.concat([nodes, ioc_nodes], ignore_index=True)

    reported = []
    for r in rels.itertuples():
        reported.append({
            "src": r.src, "dst": r.dst, "kind": "reported", "label": r.rel, "weight": 1,
            "confidence": float(r.confidence),
            "first_seen": pd.to_datetime(getattr(r, "first_seen", None), utc=True, errors="coerce"),
            "last_seen": pd.to_datetime(getattr(r, "last_seen", None), utc=True, errors="coerce"),
            "evidence": str(getattr(r, "evidence_event_id", "") or ""),
        })

    co, obs = defaultdict(list), defaultdict(list)
    ents_by_ev = mentions.groupby("event_id")["entity_id"].apply(lambda s: sorted(set(s)))
    iocs_by_ev = indicators.groupby("event_id")["value"].apply(lambda s: sorted(set(s)))
    for eid, ents in ents_by_ev.items():
        if len(ents) <= 8:
            for a, b in itertools.combinations(ents, 2):
                co[(a, b)].append(eid)
        for v in iocs_by_ev.get(eid, []):
            for a in ents:
                obs[(a, "IOC:" + v)].append(eid)

    edges = pd.DataFrame(
        reported + _aggregate(co, ev, "co-mention", "co-mentioned")
        + _aggregate(obs, ev, "observed-with", "observed with"),
        columns=EDGE_COLS)
    return nodes, edges
