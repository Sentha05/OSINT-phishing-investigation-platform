"""Stage 7 - Reporting: an evidence-linked Markdown brief with auto-generated caveats."""
import pandas as pd


def _ev_ids(res, entity_id, n=5):
    ids = res["mentions"].loc[res["mentions"]["entity_id"] == entity_id, "event_id"].tolist()
    return ", ".join(ids[:n]) + (" ..." if len(ids) > n else "")


def build_report(res, params, grid, notes=""):
    ev, qa, m = res["events"], res["qa"], res["metrics"]
    L = ["# OSINT Investigation Brief", "",
         f"*Generated {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC*", "",
         "## 1. Scope and filters", ""]
    L += [f"- {k}: {v}" for k, v in params.items()]
    L += ["", "## 2. Data quality", "",
          f"- Rows in / out: {qa['rows_in']} / {qa['rows_out']}",
          f"- Dropped for unparseable timestamps: {qa['bad_timestamps_dropped']}",
          f"- Duplicates removed: {qa['duplicates_removed']}",
          f"- Events without usable coordinates: {qa['events_without_valid_geo']} "
          f"(excluded from all maps)",
          f"- Events after filters: {len(ev)}; median source confidence "
          f"{ev['confidence'].median():.2f}" if len(ev) else "- No events after filters", ""]

    L += ["## 3. Priority entities (triage score, not a verdict)", ""]
    ents = m[~m["type"].str.startswith("indicator")].sort_values("priority", ascending=False).head(5)
    for nid, r in ents.iterrows():
        L.append(f"- **{r['name']}** ({r['type']}): degree {r['degree']}, score {r['priority']:.2f}. "
                 f"Evidence: {_ev_ids(res, nid)}")

    L += ["", "## 4. Temporal bursts", ""]
    b = res["bursts"][res["bursts"]["burst"]]
    L += [f"- {r.period:%Y-%m-%d}: {r.count} events (z = {r.z})" for r in b.itertuples()] or ["- None detected."]

    L += ["", "## 5. Geographic concentrations", ""]
    for r in grid.head(5).itertuples():
        flag = " - **SINGLE-SOURCE-DOMINATED, treat as unverified**" if r.top_source_share >= 0.6 else ""
        L.append(f"- cell ({r.lat:.0f}, {r.lon:.0f}): {r.n_events} events from {r.n_sources} sources, "
                 f"mean confidence {r.mean_conf}{flag}")

    L += ["", "## 6. Top indicators", ""]
    ind = (res["indicators"].merge(ev[["event_id", "confidence"]], on="event_id")
           .groupby(["ioc_type", "value"]).agg(events=("event_id", "nunique"), conf=("confidence", "mean"))
           .reset_index().sort_values("events", ascending=False).head(8))
    L += [f"- `{r.value}` ({r.ioc_type}): {r.events} events, mean confidence {r.conf:.2f}" for r in ind.itertuples()]

    mc = res["merge_candidates"]
    L += ["", "## 7. Limitations and caveats (auto-generated)", "",
          "- Co-mention and observed-with links are **inferred from text co-occurrence**; they are leads, not proof.",
          "- The triage score ranks where to look first; it is not a probability of involvement.",
          f"- {qa['events_without_valid_geo']} events lack coordinates, so maps under-represent them.",
          "- Heat-map intensity reflects reporting volume as well as activity; check `n_sources`."]
    if len(mc):
        L.append(f"- {len(mc)} possible duplicate entities await analyst review: "
                 + "; ".join(f"{r.name_a} / {r.name_b}" for r in mc.itertuples()))
    if notes.strip():
        L += ["", "## 8. Analyst notes", "", notes.strip()]
    return "\n".join(L)
