"""Creates a synthetic OSINT dataset in ./data (entities, relationships, events).

All actors, handles, domains and IPs are fictional (RFC 5737 documentation IPs,
.example/.test domains).  The data deliberately contains realistic problems:
alias variants, defanged indicators, duplicates, a bad timestamp, missing
coordinates, a near-duplicate entity, one noisy bot feed clustered in one city,
and a low-confidence cross-campaign link - so the visual safeguards have
something to catch.
"""
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

random.seed(42)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

entities = pd.DataFrame([
    ("E001", "Crimson Falcon", "actor", "APT-Falcon|Falcon Group"),
    ("E002", "Silent Mantis", "actor", "Mantis"),
    ("E003", "Blue Harbor", "actor", "BH-Crew"),
    ("E004", "NightLoader", "malware", "Night Loader"),
    ("E005", "RustBeacon", "malware", "Rust Beacon"),
    ("E006", "ClipSwap", "malware", ""),
    ("E007", "user_k4rma", "persona", "k4rma"),
    ("E008", "d3adpool", "persona", ""),
    ("E009", "n0va_x", "persona", "nova-x"),
    ("E010", "Meridian Bank", "organization", "Meridian"),
    ("E011", "Axis Logistics", "organization", "Axis"),
    ("E012", "Helix Health", "organization", "Helix"),
    ("E013", "Orion Telecom", "organization", "Orion"),
    ("E014", "Pioneer University", "organization", "Pioneer Univ"),
    ("E015", "Night-Loader", "malware", ""),      # near-duplicate of E004 -> merge suggestion
], columns=["entity_id", "name", "type", "aliases"])

relationships = pd.DataFrame([
    ("E001", "E004", "uses", 0.8, "2026-07-06", "2026-07-20", "EV0003"),
    ("E001", "E010", "targets", 0.7, "2026-07-06", "2026-07-20", "EV0005"),
    ("E001", "E013", "targets", 0.6, "2026-07-08", "2026-07-20", "EV0009"),
    ("E007", "E001", "operates", 0.5, "2026-07-06", "2026-07-20", "EV0011"),
    ("E002", "E005", "uses", 0.8, "2026-08-10", "2026-08-20", "EV0060"),
    ("E002", "E011", "targets", 0.7, "2026-08-10", "2026-08-20", "EV0062"),
    ("E002", "E012", "targets", 0.6, "2026-08-11", "2026-08-20", "EV0064"),
    ("E008", "E002", "operates", 0.4, "2026-08-10", "2026-08-20", "EV0066"),
    ("E003", "E006", "uses", 0.7, "2026-09-14", "2026-09-21", "EV0100"),
    ("E003", "E014", "targets", 0.6, "2026-09-14", "2026-09-21", "EV0102"),
    ("E009", "E003", "operates", 0.5, "2026-09-14", "2026-09-21", "EV0104"),
    ("E007", "E009", "collaborates-with", 0.2, "2026-09-01", "2026-09-10", "EV0090"),  # weak cross-campaign link
], columns=["src", "dst", "rel", "confidence", "first_seen", "last_seen", "evidence_event_id"])

CITY = {"Chennai": (13.08, 80.27), "Mumbai": (19.07, 72.88), "Singapore": (1.35, 103.82),
        "Frankfurt": (50.11, 8.68), "London": (51.51, -0.13), "Sao Paulo": (-23.55, -46.63),
        "Lagos": (6.52, 3.38), "Toronto": (43.65, -79.38)}
SOURCES = {"CERT advisory": ("A", [1, 2]), "Vendor blog": ("B", [2, 3]), "News wire": ("B", [2, 3, 4]),
           "Social post": ("D", [3, 4, 5]), "Paste site": ("E", [4, 5]), "Forum scrape": ("F", [3, 6]),
           "BotFeed": ("E", [4, 5])}
hexs = lambda n: "".join(random.choice("0123456789abcdef") for _ in range(n))


def ts(start, days):
    d0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    return d0 + timedelta(seconds=int(random.betavariate(1.2, 3) * days * 86400))


def defang(s):
    return s.replace(".", "[.]") if random.random() < 0.4 else s


campaigns = [
    dict(actors=["Crimson Falcon", "APT-Falcon", "Falcon Group"], mal=["NightLoader", "Night Loader", "Night-Loader"],
         persona="user_k4rma", orgs=["Meridian Bank", "Orion Telecom"],
         domains=["secure-meridian.example", "cdn-orion.test"], ips=["198.51.100.14", "198.51.100.77"],
         cities=["Frankfurt", "London", "Mumbai"], start="2026-07-06", days=14, n=42),
    dict(actors=["Silent Mantis", "Mantis"], mal=["RustBeacon", "Rust Beacon"], persona="d3adpool",
         orgs=["Axis Logistics", "Helix Health"], domains=["axis-track.example", "helix-portal.test"],
         ips=["203.0.113.21"], cities=["Singapore", "Chennai", "Toronto"], start="2026-08-10", days=10, n=34),
    dict(actors=["Blue Harbor", "BH-Crew"], mal=["ClipSwap"], persona="n0va_x", orgs=["Pioneer University"],
         domains=["pioneer-edu.example"], ips=["192.0.2.50"], cities=["Lagos", "Sao Paulo"],
         start="2026-09-14", days=7, n=24),
]
TEMPLATES = [
    "{actor} operators linked to {mal} activity; C2 traffic to {dom} ({ip}) observed against {org} in {city}.",
    "Phishing lure impersonating {org} points to {dom}; sender billing@{dom}. Attributed to {actor}.",
    "{persona} posted {mal} build details on a forum and referenced {actor}.",
    "Sandbox report: sample {sha} (md5 {md5}) contacts {ip}; family {mal}.",
    "{org} reports suspicious login attempts from {ip}; analysts suspect {actor}.",
]

rows = []


def add(when, text, city, src, etype):
    rel, creds = SOURCES[src]
    lat = lon = None
    prec = "none"
    if city and random.random() > 0.10:
        lat, lon = CITY[city]
        if random.random() < 0.7:
            lat, lon, prec = lat + random.gauss(0, 0.05), lon + random.gauss(0, 0.05), "exact"
        else:
            prec = "city"
    rows.append(dict(timestamp=when.isoformat(), source=src, source_reliability=rel,
                     info_credibility=random.choice(creds), event_type=etype, text=text,
                     location_name=city or "", lat=lat, lon=lon, geo_precision=prec))


for c in campaigns:
    for _ in range(c["n"]):
        city = random.choice(c["cities"])
        text = random.choice(TEMPLATES).format(
            actor=random.choice(c["actors"]), mal=random.choice(c["mal"]), persona=c["persona"],
            org=random.choice(c["orgs"]), city=city, dom=defang(random.choice(c["domains"])),
            ip=defang(random.choice(c["ips"])), sha=hexs(64), md5=hexs(32))
        src = random.choice(["CERT advisory", "Vendor blog", "News wire", "Social post", "Paste site", "Forum scrape"])
        add(ts(c["start"], c["days"]), text, city, src,
            random.choice(["phishing", "c2-beacon", "credential-theft", "malware-sample"]))

for _ in range(40):                                   # background noise
    org, ip = random.choice(entities[entities.type == "organization"].name.tolist()), "198.51.100.%d" % random.randint(100, 250)
    add(ts("2026-07-01", 90), f"Port scan observed against {org} from {ip}.", random.choice(list(CITY)),
        random.choice(["News wire", "Social post", "Paste site"]), "scan")

for _ in range(30):                                   # one noisy bot feed, one city
    add(ts("2026-08-01", 40), f"Automated alert: scan against Helix Health from 192.0.2.{random.randint(60, 99)}.",
        "Toronto", "BotFeed", "scan")

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df.insert(0, "event_id", [f"EV{i + 1:04d}" for i in range(len(df))])
df = pd.concat([df, df.iloc[[5]]], ignore_index=True)                       # exact duplicate
bad = df.iloc[[7]].copy()
bad["event_id"], bad["timestamp"] = "EV9999", "not-a-date"                  # bad timestamp
df = pd.concat([df, bad], ignore_index=True)

def names_of(eid):
    r = entities[entities.entity_id == eid].iloc[0]
    return [r["name"].lower()] + [a.lower() for a in r["aliases"].split("|") if a]


def find_evidence(src, dst):                     # first real event that mentions both ends
    a, b = names_of(src), names_of(dst)
    for r in df.itertuples():
        t = str(r.text).lower()
        if any(x in t for x in a) and any(x in t for x in b):
            return r.event_id
    return ""


relationships["evidence_event_id"] = [find_evidence(r.src, r.dst) for r in relationships.itertuples()]

entities.to_csv(OUT / "entities.csv", index=False)
relationships.to_csv(OUT / "relationships.csv", index=False)
df.to_csv(OUT / "events.csv", index=False)
print(f"Wrote {len(entities)} entities, {len(relationships)} relationships, {len(df)} events to {OUT}")
