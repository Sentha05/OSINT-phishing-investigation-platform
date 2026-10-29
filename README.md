# Visualization-Based OSINT Intelligence Platform (prototype)

Pipeline: acquisition -> pre-processing -> entity/IOC extraction -> correlation -> analytics -> visualization -> reporting.

## Run (Windows / macOS / Linux) - Python 3.10+
```
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python generate_sample_data.py          # creates ./data (synthetic)
streamlit run app.py                    # opens http://localhost:8501
```
Map tiles need an internet connection (OpenStreetMap). Everything else runs offline.

## Your own data (CSV)
- entities.csv: entity_id, name, type, aliases (separate aliases with |)
- relationships.csv: src, dst, rel, confidence (0-1), first_seen, last_seen, evidence_event_id
- events.csv: event_id, timestamp, source, text, source_reliability (A-F), info_credibility (1-6),
  event_type, location_name, lat, lon, geo_precision

## Layout
app.py (UI) | osint/ingest.py | preprocess.py | extract.py | correlate.py | analytics.py | viz.py | report.py | pipeline.py
