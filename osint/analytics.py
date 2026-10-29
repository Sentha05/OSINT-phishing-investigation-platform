"""Stage 5 - Analytics: graph metrics, communities, temporal bursts, geo grid, triage score."""
import networkx as nx
import numpy as np
import pandas as pd


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for n in nodes.itertuples():
        G.add_node(n.node_id, name=n.name, type=n.type)
    for e in edges.itertuples():
        if e.src not in G or e.dst not in G:
            continue
        if G.has_edge(e.src, e.dst):
            d = G[e.src][e.dst]
            d["kinds"].add(e.kind)
            d["weight"] += e.weight
            d["confidence"] = max(d["confidence"], e.confidence)
        else:
            G.add_edge(e.src, e.dst, kinds={e.kind}, weight=e.weight, confidence=e.confidence)
    return G


def _norm(s: pd.Series) -> pd.Series:
    m = s.max()
    return s / m if m and m > 0 else s * 0.0


def node_metrics(G: nx.Graph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["name", "type", "degree", "betweenness",
                                     "community", "ioc_links", "priority"])
    try:
        comms = nx.community.louvain_communities(G, weight="weight", seed=7)
    except AttributeError:
        comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
    cmap = {n: i for i, c in enumerate(comms) for n in c}
    bet = nx.betweenness_centrality(G) if G.number_of_nodes() < 400 else {}
    df = pd.DataFrame({
        "name": [G.nodes[n]["name"] for n in G],
        "type": [G.nodes[n]["type"] for n in G],
        "degree": [G.degree(n) for n in G],
        "betweenness": [bet.get(n, 0.0) for n in G],
        "community": [cmap.get(n, -1) for n in G],
        "ioc_links": [sum(G.nodes[m]["type"].startswith("indicator") for m in G[n]) for n in G],
    }, index=pd.Index(list(G.nodes), name="node_id"))
    # Triage score: a ranking aid for where to look first - NOT a probability of guilt.
    df["priority"] = (0.5 * _norm(df["betweenness"]) + 0.3 * _norm(df["degree"].astype(float))
                      + 0.2 * _norm(df["ioc_links"].astype(float))).round(3)
    return df


def time_bursts(events: pd.DataFrame, z_thr: float = 2.0) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["period", "count", "z", "burst"])
    c = events.set_index("timestamp").resample("D").size().rename("count").reset_index()
    c = c.rename(columns={"timestamp": "period"})
    sd = c["count"].std(ddof=0)
    c["z"] = ((c["count"] - c["count"].mean()) / sd).round(2) if sd > 0 else 0.0
    c["burst"] = c["z"] >= z_thr
    return c


def geo_grid(events: pd.DataFrame, cell: float = 2.0, weight_by_source: bool = True) -> pd.DataFrame:
    ev = events[events["has_geo"]].copy()
    cols = ["lat", "lon", "weight", "n_events", "n_sources", "mean_conf", "top_source_share"]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    ev["clat"] = (np.floor(ev["lat"] / cell) * cell + cell / 2).round(3)
    ev["clon"] = (np.floor(ev["lon"] / cell) * cell + cell / 2).round(3)
    # Down-weight prolific sources so one noisy feed cannot dominate the heat map.
    per_source = ev["source"].map(ev["source"].value_counts())
    ev["w"] = (1.0 / per_source) if weight_by_source else 1.0
    g = ev.groupby(["clat", "clon"]).agg(
        weight=("w", "sum"), n_events=("event_id", "count"), n_sources=("source", "nunique"),
        mean_conf=("confidence", "mean"),
        top_source_share=("source", lambda s: s.value_counts(normalize=True).iloc[0]),
    ).reset_index().rename(columns={"clat": "lat", "clon": "lon"})
    g["mean_conf"] = g["mean_conf"].round(2)
    g["top_source_share"] = g["top_source_share"].round(2)
    return g.sort_values("weight", ascending=False)[cols]
