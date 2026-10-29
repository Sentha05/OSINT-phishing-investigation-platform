"""Orchestrates stages 1-5 so the UI (or a unit test) can call a single function."""
import pandas as pd
from . import preprocess, extract, correlate, analytics


def run(entities, rels, events_raw, start=None, end=None, min_conf=0.0, sources=None, min_edge_conf=0.0):
    events, qa = preprocess.clean_events(events_raw)
    if start is not None:
        events = events[events["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        events = events[events["timestamp"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    if min_conf > 0:
        events = events[events["confidence"] >= min_conf]
    if sources:
        events = events[events["source"].isin(sources)]
    events = events.reset_index(drop=True)

    mentions = extract.link_entities(events, extract.build_alias_index(entities))
    indicators = extract.extract_indicators(events)
    nodes, edges = correlate.build_graph_tables(entities, rels, events, mentions, indicators)
    edges = edges[edges["confidence"] >= min_edge_conf].reset_index(drop=True)
    G = analytics.build_graph(nodes, edges)
    return {
        "events": events, "qa": qa, "mentions": mentions, "indicators": indicators,
        "nodes": nodes, "edges": edges, "graph": G,
        "metrics": analytics.node_metrics(G),
        "bursts": analytics.time_bursts(events),
        "merge_candidates": correlate.merge_candidates(entities),
    }
