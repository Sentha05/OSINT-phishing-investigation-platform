"""Stage 6 - Visualization (Plotly).  Each view is chosen for a specific investigative question:

  link chart  - WHO is connected to WHAT, and how strong is the evidence?
  matrix      - dense networks where node-link views become a hairball
  timeline    - WHEN, and in what order / are there bursts?
  heat map    - WHERE is activity concentrated (and is it just one noisy source)?

Misinterpretation safeguards are encoded visually: edge style = evidence kind,
edge opacity = confidence, marker size/colour = confidence, and every view
carries a reminder of what its positions do and do not mean.
"""
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EDGE_STYLE = {
    "reported": dict(color="#1f77b4", dash="solid"),
    "co-mention": dict(color="#7f7f7f", dash="dot"),
    "observed-with": dict(color="#d62728", dash="dash"),
}
TYPE_COLOR = {"actor": "#d62728", "malware": "#9467bd", "persona": "#ff7f0e", "organization": "#2ca02c"}


def _empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=400)
    return fig


def link_chart(G, edges, metrics, kinds, min_conf=0.0, color_by="type", focus=None, radius=2, show_ioc=True):
    if focus and focus in G:
        keep = set(nx.ego_graph(G, focus, radius=radius).nodes)
    else:
        keep = set(G.nodes)
    e = edges[edges["kind"].isin(kinds) & (edges["confidence"] >= min_conf)
              & edges["src"].isin(keep) & edges["dst"].isin(keep)]
    if not show_ioc:
        e = e[~e["src"].str.startswith("IOC:") & ~e["dst"].str.startswith("IOC:")]
    if e.empty:
        return _empty("No links match the current filters.")

    L = nx.Graph()
    L.add_edges_from(zip(e["src"], e["dst"]))
    pos = nx.spring_layout(L, seed=7, k=1.5 / np.sqrt(max(len(L), 1)), iterations=120)

    fig = go.Figure()
    e = e.assign(band=np.where(e["confidence"] >= 0.6, "high", "low"))
    for (kind, band), g in e.groupby(["kind", "band"]):
        xs, ys = [], []
        for r in g.itertuples():
            (x0, y0), (x1, y1) = pos[r.src], pos[r.dst]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        st = EDGE_STYLE.get(kind, dict(color="#999", dash="solid"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                 opacity=0.9 if band == "high" else 0.3,
                                 line=dict(width=2.2 if band == "high" else 1, color=st["color"], dash=st["dash"]),
                                 name=f"{kind} - {band} confidence"))
    mid_x = [(pos[r.src][0] + pos[r.dst][0]) / 2 for r in e.itertuples()]
    mid_y = [(pos[r.src][1] + pos[r.dst][1]) / 2 for r in e.itertuples()]
    hover = [f"<b>{r.kind}</b>: {r.label}<br>confidence {r.confidence:.2f} | weight {r.weight}"
             f"<br>evidence: {str(r.evidence)[:120]}" for r in e.itertuples()]
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", marker=dict(size=9, opacity=0),
                             hovertext=hover, hoverinfo="text", showlegend=False))

    nd = metrics.loc[[n for n in L.nodes if n in metrics.index]].copy()
    pal = px.colors.qualitative.Set2
    nd["group"] = nd["type"] if color_by == "type" else ("community " + nd["community"].astype(str))
    for grp, g in nd.groupby("group"):
        base = grp.split(":")[0]
        color = TYPE_COLOR.get(grp, "#17becf") if color_by == "type" else pal[int(grp.split()[-1]) % len(pal)]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in g.index], y=[pos[n][1] for n in g.index],
            mode="markers+text", name=grp,
            text=[nm if not t.startswith("indicator") else "" for nm, t in zip(g["name"], g["type"])],
            textposition="top center",
            marker=dict(size=(10 + 3 * g["degree"].clip(upper=10)).tolist(), color=color,
                        symbol="diamond" if base == "indicator" else "circle",
                        line=dict(width=1, color="white")),
            hovertext=[f"<b>{nm}</b><br>{t}<br>degree {d} | betweenness {b:.3f}<br>triage score {p:.2f}"
                       for nm, t, d, b, p in zip(g["name"], g["type"], g["degree"], g["betweenness"], g["priority"])],
            hoverinfo="text"))
    fig.update_layout(height=680, margin=dict(l=0, r=0, t=40, b=0),
                      title="Link chart - position carries no meaning; read edge style (kind) and opacity (confidence)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h", y=-0.02))
    return fig


def matrix_view(G, metrics, top_n=60):
    if G.number_of_nodes() == 0:
        return _empty("No data.")
    m = metrics.sort_values(["community", "degree"], ascending=[True, False])
    top = metrics.sort_values("degree", ascending=False).head(top_n).index
    order = [n for n in m.index if n in set(top)]
    names = [G.nodes[n]["name"] for n in order]
    z = nx.to_numpy_array(G, nodelist=order, weight="weight")
    fig = go.Figure(go.Heatmap(z=z, x=names, y=names, colorscale="Blues",
                               hovertemplate="%{y} - %{x}<br>weight %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=40, b=0),
                      title=f"Adjacency matrix of top {len(order)} nodes, grouped by community (no edge crossings)")
    return fig


def timeline(events, bursts):
    if events.empty:
        return _empty("No events in the selected range.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.28, 0.72], vertical_spacing=0.04)
    fig.add_trace(go.Bar(x=bursts["period"], y=bursts["count"],
                         marker_color=np.where(bursts["burst"], "#d62728", "#9aa5b1"),
                         name="events/day (red = burst, z>=2)",
                         hovertemplate="%{x|%Y-%m-%d}<br>%{y} events<extra></extra>"), row=1, col=1)
    hover = [f"{r.event_id} | {r.source} | conf {r.confidence:.2f}<br>{str(r.text_clean)[:140]}"
             for r in events.itertuples()]
    fig.add_trace(go.Scatter(
        x=events["timestamp"], y=events["event_type"], mode="markers", hovertext=hover, hoverinfo="text",
        marker=dict(size=(6 + events["confidence"] * 14).tolist(), color=events["confidence"], cmin=0, cmax=1,
                    colorscale="Viridis", colorbar=dict(title="confidence"), line=dict(width=0.5, color="white")),
        name="events"), row=2, col=1)
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                      title="Timeline - all times UTC; marker size and colour = source confidence")
    return fig


def _center(grid_or_ev):
    return dict(lat=float(grid_or_ev["lat"].mean()), lon=float(grid_or_ev["lon"].mean()))


def heatmap(grid, radius=35):
    if grid.empty:
        return _empty("No events with valid coordinates.")
    kw = dict(lat="lat", lon="lon", z="weight", radius=radius, center=_center(grid), zoom=1.2, height=620,
              hover_data={"n_events": True, "n_sources": True, "mean_conf": True, "top_source_share": True})
    if hasattr(px, "density_map"):
        fig = px.density_map(grid, map_style="open-street-map", **kw)
    else:
        fig = px.density_mapbox(grid, mapbox_style="open-street-map", **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


def points_map(events):
    ev = events[events["has_geo"]]
    if ev.empty:
        return _empty("No events with valid coordinates.")
    kw = dict(lat="lat", lon="lon", color="geo_precision", hover_name="event_id",
              hover_data={"source": True, "confidence": True, "lat": False, "lon": False},
              center=_center(ev), zoom=1.2, height=620)
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(ev, map_style="open-street-map", **kw)
    else:
        fig = px.scatter_mapbox(ev, mapbox_style="open-street-map", **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig
