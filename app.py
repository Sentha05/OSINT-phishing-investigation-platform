"""Streamlit front-end: run with   streamlit run app.py"""
from pathlib import Path
import pandas as pd
import streamlit as st

from osint import analytics, ingest, pipeline, report, viz

st.set_page_config(page_title="OSINT Visual Intelligence Platform", layout="wide")
st.title("Visualization-Based OSINT Intelligence Platform")
st.caption("Prototype - the bundled dataset is synthetic. Links marked 'co-mention' or 'observed-with' are inferred leads, not proof.")

# ------------------------------------------------------------------ 1. acquisition
DATA_DIR = Path(__file__).parent / "data"
mode = st.sidebar.radio("Data source", ["Bundled sample", "Upload CSVs"])
try:
    if mode == "Bundled sample":
        entities, rels, events_raw = ingest.load_folder(DATA_DIR)
    else:
        f1 = st.sidebar.file_uploader("entities.csv", type="csv")
        f2 = st.sidebar.file_uploader("relationships.csv", type="csv")
        f3 = st.sidebar.file_uploader("events.csv", type="csv")
        if not (f1 and f2 and f3):
            st.info("Upload all three CSV files to continue.")
            st.stop()
        entities, rels, events_raw = ingest.load_uploads(f1, f2, f3)
except FileNotFoundError:
    st.error("Sample data not found. Run  `python generate_sample_data.py`  once, then reload.")
    st.stop()
except ValueError as err:
    st.error(str(err))
    st.stop()

# ------------------------------------------------------------------ filters
ts = pd.to_datetime(events_raw["timestamp"], utc=True, errors="coerce").dropna()
d_min, d_max = ts.min().date(), ts.max().date()
date_range = st.sidebar.date_input("Date range (UTC)", (d_min, d_max), min_value=d_min, max_value=d_max)
if len(date_range) != 2:
    st.stop()
min_conf = st.sidebar.slider("Min event confidence", 0.0, 1.0, 0.0, 0.05,
                             help="Source reliability x information credibility (Admiralty grading).")
min_edge = st.sidebar.slider("Min edge confidence", 0.0, 1.0, 0.0, 0.05)
all_sources = sorted(events_raw["source"].dropna().unique())
sources = st.sidebar.multiselect("Sources", all_sources, default=all_sources)


@st.cache_data(show_spinner="Running pipeline...")
def run_pipeline(entities, rels, events_raw, start, end, min_conf, sources, min_edge):
    return pipeline.run(entities, rels, events_raw, start, end, min_conf, list(sources), min_edge)


res = run_pipeline(entities, rels, events_raw, date_range[0], date_range[1], min_conf, tuple(sources), min_edge)
events, G, metrics, edges = res["events"], res["graph"], res["metrics"], res["edges"]
if events.empty:
    st.warning("No events match the filters.")
    st.stop()

tabs = st.tabs(["Overview", "Link chart", "Matrix", "Timeline", "Geo heat map", "Evidence", "Report"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Events", len(events))
    c[1].metric("Entities", int((~metrics["type"].str.startswith("indicator")).sum()))
    c[2].metric("Indicators", int(metrics["type"].str.startswith("indicator").sum()))
    c[3].metric("Links", len(edges))
    c[4].metric("With valid geo", f"{events['has_geo'].mean():.0%}")
    st.subheader("Pre-processing log")
    st.dataframe(pd.DataFrame([res["qa"]]), hide_index=True)
    st.subheader("Source mix (a view dominated by one source is a warning sign)")
    st.bar_chart(events["source"].value_counts())
    if len(res["merge_candidates"]):
        st.subheader("Possible duplicate entities - analyst decision required")
        st.dataframe(res["merge_candidates"], hide_index=True)

# ------------------------------------------------------------------ link chart
named = metrics[~metrics["type"].str.startswith("indicator")]
with tabs[1]:
    a, b, c2, d = st.columns(4)
    kinds = a.multiselect("Link kinds", ["reported", "co-mention", "observed-with"],
                          default=["reported", "co-mention", "observed-with"])
    color_by = b.radio("Colour by", ["type", "community"], horizontal=True)
    focus_name = c2.selectbox("Focus on entity (ego network)", ["(whole graph)"] + sorted(named["name"]))
    show_ioc = d.checkbox("Show indicator nodes", True)
    focus = None if focus_name == "(whole graph)" else named.index[named["name"] == focus_name][0]
    st.plotly_chart(viz.link_chart(G, edges, metrics, kinds, min_edge, color_by, focus, show_ioc=show_ioc),
                    width="stretch")
    st.caption("Solid = reported by a source | dotted = co-mention (inferred) | dashed red = entity seen with indicator (inferred). "
               "Faint lines = confidence < 0.6.")
    st.subheader("Triage ranking")
    st.dataframe(named.sort_values("priority", ascending=False)[["name", "type", "degree", "betweenness", "community", "priority"]]
                 .head(15), width="stretch")

with tabs[2]:
    st.plotly_chart(viz.matrix_view(G, metrics), width="stretch")
    st.caption("Use the matrix when the link chart becomes a hairball. Blocks along the diagonal are communities.")

# ------------------------------------------------------------------ timeline
with tabs[3]:
    f = st.selectbox("Show only events mentioning", ["(all events)"] + sorted(named["name"]), key="tl")
    ev_t = events
    if f != "(all events)":
        eid = named.index[named["name"] == f][0]
        ev_t = events[events["event_id"].isin(res["mentions"].loc[res["mentions"]["entity_id"] == eid, "event_id"])]
    st.plotly_chart(viz.timeline(ev_t, analytics.time_bursts(ev_t)), width="stretch")

# ------------------------------------------------------------------ geo
with tabs[4]:
    a, b, c3 = st.columns(3)
    view = a.radio("View", ["Heat map", "Points"], horizontal=True)
    cell = b.slider("Grid cell (degrees)", 0.5, 10.0, 2.0, 0.5)
    wsrc = c3.checkbox("Down-weight prolific sources", True,
                       help="Stops a single noisy feed from dominating the map.")
    grid = analytics.geo_grid(events, cell, wsrc)
    st.plotly_chart(viz.heatmap(grid) if view == "Heat map" else viz.points_map(events), width="stretch")
    miss = int((~events["has_geo"]).sum())
    st.caption(f"{miss} of {len(events)} events have no valid coordinates and are NOT shown. "
               "Intensity reflects reporting volume as well as real activity.")
    st.dataframe(grid.head(15), hide_index=True)

# ------------------------------------------------------------------ evidence
with tabs[5]:
    q = st.text_input("Search event text / indicator")
    ev_e = events[events["text_clean"].str.contains(q, case=False, regex=False)] if q else events
    st.dataframe(ev_e[["event_id", "timestamp", "source", "source_reliability", "info_credibility",
                       "confidence", "event_type", "location_name", "geo_precision", "text_clean"]],
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ report
with tabs[6]:
    notes = st.text_area("Analyst notes (included in the report)", height=120)
    params = {"Date range": f"{date_range[0]} to {date_range[1]}", "Min event confidence": min_conf,
              "Min edge confidence": min_edge, "Sources": ", ".join(sources)}
    md = report.build_report(res, params, analytics.geo_grid(events, 2.0, True), notes)
    st.markdown(md)
    st.download_button("Download report (.md)", md, file_name="osint_brief.md", mime="text/markdown")
