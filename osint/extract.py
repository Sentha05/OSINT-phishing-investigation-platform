"""Stage 3 - Entity and indicator extraction.

* Indicators of compromise (IP, domain, hash, e-mail, CVE) - regular expressions.
* Entities - dictionary/alias matching against the entity register.
* Optional spaCy NER proposes *candidate* entities that are never auto-linked.
"""
import re
import pandas as pd

IOC = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I),
    "domain": re.compile(
        r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ru|cn|info|xyz|top|biz|co|in|dev|site|example|test)\b",
        re.I),
}


def extract_indicators(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in events.itertuples():
        text = r.text_clean
        for kind in ("email", "ipv4", "sha256", "md5", "cve"):
            for m in IOC[kind].findall(text):
                rows.append((r.event_id, kind, m.lower()))
        stripped = IOC["email"].sub(" ", text)          # avoid e-mail domains counted twice
        for m in IOC["domain"].findall(stripped):
            rows.append((r.event_id, "domain", m.lower()))
    return pd.DataFrame(rows, columns=["event_id", "ioc_type", "value"]).drop_duplicates()


def build_alias_index(entities: pd.DataFrame):
    rows = []
    for e in entities.itertuples():
        names = [str(e.name)]
        raw = getattr(e, "aliases", "")
        if isinstance(raw, str) and raw.strip():
            names += [a.strip() for a in raw.split("|") if a.strip()]
        rows += [(n, e.entity_id) for n in names]
    rows.sort(key=lambda x: -len(x[0]))                  # longest alias first
    return [(re.compile(r"(?<!\w)" + re.escape(n) + r"(?!\w)", re.I), eid, n) for n, eid in rows]


def link_entities(events: pd.DataFrame, alias_index) -> pd.DataFrame:
    rows = []
    for r in events.itertuples():
        found = set()
        for pat, eid, name in alias_index:
            if eid not in found and pat.search(r.text_clean):
                found.add(eid)
                rows.append((r.event_id, eid, name, "dictionary"))
    return pd.DataFrame(rows, columns=["event_id", "entity_id", "matched_text", "method"])


def spacy_candidates(events: pd.DataFrame) -> pd.DataFrame:
    """Optional: returns ORG/PERSON/GPE strings spaCy finds. Empty if spaCy is absent."""
    cols = ["event_id", "text", "label"]
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return pd.DataFrame(columns=cols)
    out = []
    for r in events.itertuples():
        for ent in nlp(r.text_clean).ents:
            if ent.label_ in {"ORG", "PERSON", "GPE"}:
                out.append((r.event_id, ent.text, ent.label_))
    return pd.DataFrame(out, columns=cols)
